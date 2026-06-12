from __future__ import annotations

import copy
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from genode.canonical_experiment_layout import (
    AVERAGED_SCHEDULE_COMPONENTS,
    CANONICAL_CONTEXT_SAMPLE_COUNT,
    CANONICAL_SUPERVISION_SCHEDULE_KEYS,
    REVERSED_SCHEDULE_BASE,
    schedule_family_for_key,
)
from genode.gipo.density_representation import (
    DEFAULT_DENSITY_BIN_COUNT,
    DENSITY_PROTOCOL,
    average_density_masses,
    density_log_features,
    density_mass_hash,
    density_mass_to_time_grid,
    density_metadata,
    grid_to_density_mass,
    reference_grid_hash,
    uniform_reference_grid,
    validate_reference_grid,
)
from genode.gipo.models import (
    SETTING_ENCODER_MODE_CONTINUOUS_V3,
    SettingEncoderConfig,
    setting_encoder_config_from_payload,
    setting_features,
    solver_macro_steps,
    validate_setting_feature_mode,
    validate_time_grid,
)
from genode.gipo.objectives import (
    CONDITIONAL_METRIC_SPECS,
    DEFAULT_REWARD_EPS,
    FORECAST_METRIC_SPECS,
    MOLECULE_METRIC_SPECS,
    MetricObjectiveSpec,
    UNIFORM_SCHEDULE_KEY,
    objective_weight_map_for_keys,
    uniform_anchored_objective_columns,
)
from genode.gipo.schedule_hash import json_hash as _canonical_json_hash
from genode.gipo.schedule_hash import schedule_grid_hash
from genode.solver_protocol import normalize_solver_key
from genode.gipo.ser_ptg_reference import SER_PTG_SCHEDULE_KEY
from genode.schedule_transfer.diffusion_flow_schedules import (
    BASELINE_SCHEDULE_KEYS,
    EXPERIMENTAL_FIXED_SCHEDULE_KEYS,
    build_schedule_grid,
)

MetricRow = Mapping[str, Any]
ContextPairKey = Tuple[str, str, int, str, int | None]
ScheduleGridKey = Tuple[str, str, int] | Tuple[str, str, int, int]

GIPO_PROTOCOL = "gipo_density"
DEFAULT_SUPERVISION_SCHEDULE_KEYS: Tuple[str, ...] = CANONICAL_SUPERVISION_SCHEDULE_KEYS
DEFAULT_SUPPORT_SCHEDULE_KEYS: Tuple[str, ...] = DEFAULT_SUPERVISION_SCHEDULE_KEYS
GIPO_SUPPORT_SCHEDULE_KEYS: Tuple[str, ...] = DEFAULT_SUPERVISION_SCHEDULE_KEYS
EXPERIMENTAL_SUPERVISION_SCHEDULE_KEYS: Tuple[str, ...] = DEFAULT_SUPERVISION_SCHEDULE_KEYS
DEFAULT_CONTEXT_CALIBRATION_TOTAL = CANONICAL_CONTEXT_SAMPLE_COUNT
DEFAULT_CONTEXT_CALIBRATION_VALIDATION_FRACTION = 0.20
MIN_CONTEXT_CALIBRATION_TOTAL = 72
MAX_CONTEXT_CALIBRATION_TOTAL = CANONICAL_CONTEXT_SAMPLE_COUNT
DEFAULT_TEACHER_TARGET_TEMPERATURE = 0.05
STUDENT_TARGET_PROTOCOL_SOFT_MIXTURE = "teacher_weighted_soft_mixture"
MODEL_PAYLOAD_VERSION = 4
ARCHITECTURE_DENSITY_FORM_TRANSFORMER = "density_form_transformer"
ARCHITECTURE_DENSITY_QUERY_TRANSFORMER = "density_query_transformer"
CONDITIONING_STYLE_ADDITIVE_MLP = "additive_mlp"
DENSITY_TOKEN_ATTENTION_ROPE = "bin_self_attention_rope"
TEACHER_OUTPUT_METRIC_VECTOR = "metric_vector"
TEACHER_SCALARIZATION_WEIGHTED_AVERAGE = "weighted_metric_average"
TEACHER_METRIC_TARGET_PROTOCOL_VECTOR = "family_metric_utility_vector"
TEACHER_METRIC_MASK_PROTOCOL = "row_valid_component_mask"
DEFAULT_TEACHER_METRIC_TARGET_KEYS: Tuple[str, ...] = ("u_comp_uniform",)
TEACHER_METRIC_TARGET_KEYS: Tuple[str, ...] = DEFAULT_TEACHER_METRIC_TARGET_KEYS
TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET = "weighted_normalized_regret"
DEFAULT_TEACHER_CHECKPOINT_SELECTION_MODE = TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET
STUDENT_CHECKPOINT_SELECTION_VALIDATION_CE = "validation_ce"
DEFAULT_STUDENT_CHECKPOINT_SELECTION_MODE = STUDENT_CHECKPOINT_SELECTION_VALIDATION_CE
DEFAULT_TEACHER_SELECTION_COMPONENT_WEIGHTS: Dict[str, float] = {
    "context": 0.25,
    "density_family": 0.25,
    "unseen_nfe": 0.50,
}
DEFAULT_DENSITY_FAMILY_HOLDOUT_SCHEDULE_KEYS: Tuple[str, ...] = (
    "flowts_power_sampling",
    "flowts_power_sampling_reversed",
    "flowts_power_sampling_avg_reversed",
)
DEFAULT_TRANSFORMER_HIDDEN_DIM = 64
DEFAULT_TRANSFORMER_LAYERS = 2
DEFAULT_TRANSFORMER_HEADS = 4
DEFAULT_TRANSFORMER_DROPOUT = 0.05
SERIES_CONDITIONING_NONE_CONTEXT_ONLY = "none_context_only"
SERIES_CONDITIONING_DIM = 0


def validate_gipo_architecture(value: str, *, role: str | None = None) -> str:
    default = ARCHITECTURE_DENSITY_FORM_TRANSFORMER if role == "teacher" else ARCHITECTURE_DENSITY_QUERY_TRANSFORMER
    arch = str(value).strip() or default
    expected = {
        "teacher": ARCHITECTURE_DENSITY_FORM_TRANSFORMER,
        "student": ARCHITECTURE_DENSITY_QUERY_TRANSFORMER,
    }.get(str(role or ""), "")
    if expected:
        if arch != expected:
            raise ValueError(f"GIPO {role} architecture must be {expected!r}, got {value!r}.")
        return arch
    allowed = {ARCHITECTURE_DENSITY_FORM_TRANSFORMER, ARCHITECTURE_DENSITY_QUERY_TRANSFORMER}
    if arch not in allowed:
        raise ValueError(f"GIPO architecture must be one of {sorted(allowed)}, got {value!r}.")
    return arch


def validate_canonical_conditioning_style(
    model_config: Mapping[str, Any] | None,
    *,
    require_present: bool = False,
) -> str:
    if not model_config or "conditioning_style" not in model_config:
        if require_present:
            raise ValueError(
                f"GIPO density transformer checkpoints require conditioning_style={CONDITIONING_STYLE_ADDITIVE_MLP!r}."
            )
        return CONDITIONING_STYLE_ADDITIVE_MLP
    style = str(model_config["conditioning_style"]).strip()
    if style == CONDITIONING_STYLE_ADDITIVE_MLP:
        return style
    raise ValueError(
        f"GIPO density transformers require conditioning_style={CONDITIONING_STYLE_ADDITIVE_MLP!r}; "
        f"got {style!r}."
    )


def validate_gipo_teacher_training_metadata(metadata: Mapping[str, Any] | None) -> Dict[str, Any]:
    teacher_training = dict(metadata or {})
    teacher_target = str(teacher_training.get("teacher_target", ""))
    if teacher_target not in {"metric_vector", "metric_vector_uniform"}:
        raise ValueError("GIPO checkpoint must come from a metric-vector teacher.")
    teacher_metric_targets = validate_teacher_metric_target_keys(teacher_training.get("teacher_metric_targets", ()))
    if str(teacher_training.get("teacher_scalarization", "")) != TEACHER_SCALARIZATION_WEIGHTED_AVERAGE:
        raise ValueError(
            f"GIPO checkpoint teacher_scalarization must be {TEACHER_SCALARIZATION_WEIGHTED_AVERAGE!r}."
        )
    selection = dict(teacher_training.get("teacher_checkpoint_selection", {}) or {})
    if str(selection.get("selection_protocol", "")) != TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET:
        raise ValueError(
            f"GIPO checkpoints must use teacher checkpoint selection "
            f"{TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET!r}."
        )
    if bool(selection.get("uses_validation_labels", False)):
        raise ValueError("GIPO teacher checkpoint selection metadata must not use validation labels.")
    return {
        "teacher_target": teacher_target,
        "teacher_metric_targets": teacher_metric_targets,
        "teacher_metric_target_protocol": str(
            teacher_training.get("teacher_metric_target_protocol", TEACHER_METRIC_TARGET_PROTOCOL_VECTOR)
        ),
        "teacher_metric_mask_protocol": str(
            teacher_training.get("teacher_metric_mask_protocol", TEACHER_METRIC_MASK_PROTOCOL)
        ),
        "teacher_scalarization": TEACHER_SCALARIZATION_WEIGHTED_AVERAGE,
        "teacher_checkpoint_selection": selection,
    }


def validate_gipo_density_token_attention(model_config: Mapping[str, Any] | None, *, require_present: bool = False) -> str:
    if not model_config or "density_token_attention" not in model_config:
        if require_present:
            raise ValueError(f"GIPO density transformer checkpoints require density_token_attention={DENSITY_TOKEN_ATTENTION_ROPE!r}.")
        return DENSITY_TOKEN_ATTENTION_ROPE
    attention = str(model_config["density_token_attention"]).strip()
    if attention != DENSITY_TOKEN_ATTENTION_ROPE:
        raise ValueError(
            f"GIPO density transformers require density_token_attention={DENSITY_TOKEN_ATTENTION_ROPE!r}; "
            f"got {attention!r}."
        )
    return attention


def validate_gipo_teacher_output(model_config: Mapping[str, Any] | None, *, require_present: bool = False) -> str:
    if not model_config or "teacher_output" not in model_config:
        if require_present:
            raise ValueError(f"GIPO teacher checkpoints require teacher_output={TEACHER_OUTPUT_METRIC_VECTOR!r}.")
        return TEACHER_OUTPUT_METRIC_VECTOR
    output = str(model_config["teacher_output"]).strip()
    if output != TEACHER_OUTPUT_METRIC_VECTOR:
        raise ValueError(f"GIPO teacher output must be {TEACHER_OUTPUT_METRIC_VECTOR!r}; got {output!r}.")
    if require_present and "teacher_metric_targets" not in model_config:
        raise ValueError("GIPO teacher checkpoints require explicit teacher_metric_targets.")
    validate_teacher_metric_target_keys(model_config.get("teacher_metric_targets", TEACHER_METRIC_TARGET_KEYS))
    scalarization = str(model_config.get("teacher_scalarization", TEACHER_SCALARIZATION_WEIGHTED_AVERAGE))
    if scalarization != TEACHER_SCALARIZATION_WEIGHTED_AVERAGE:
        raise ValueError(
            f"GIPO teacher scalarization must be {TEACHER_SCALARIZATION_WEIGHTED_AVERAGE!r}; got {scalarization!r}."
        )
    target_protocol = str(model_config.get("teacher_metric_target_protocol", TEACHER_METRIC_TARGET_PROTOCOL_VECTOR))
    if target_protocol != TEACHER_METRIC_TARGET_PROTOCOL_VECTOR:
        raise ValueError(
            f"GIPO teacher metric target protocol must be {TEACHER_METRIC_TARGET_PROTOCOL_VECTOR!r}; got {target_protocol!r}."
        )
    mask_protocol = str(model_config.get("teacher_metric_mask_protocol", TEACHER_METRIC_MASK_PROTOCOL))
    if mask_protocol != TEACHER_METRIC_MASK_PROTOCOL:
        raise ValueError(
            f"GIPO teacher metric mask protocol must be {TEACHER_METRIC_MASK_PROTOCOL!r}; got {mask_protocol!r}."
        )
    return output


def validate_gipo_attention_heads(attention_heads: int) -> int:
    heads = int(attention_heads)
    if heads != DEFAULT_TRANSFORMER_HEADS:
        raise ValueError(f"GIPO density transformers require attention_heads={DEFAULT_TRANSFORMER_HEADS}; got {heads}.")
    return heads


def validate_series_conditioning(value: str | None = None) -> str:
    conditioning = str(value or SERIES_CONDITIONING_NONE_CONTEXT_ONLY).strip()
    if conditioning != SERIES_CONDITIONING_NONE_CONTEXT_ONLY:
        raise ValueError(
            f"GIPO density transformers require series_conditioning={SERIES_CONDITIONING_NONE_CONTEXT_ONLY!r}; "
            f"got {conditioning!r}."
        )
    return conditioning


def _normalized_metric_weights(crps_weight: float, mase_weight: float) -> Tuple[float, float]:
    crps = float(crps_weight)
    mase = float(mase_weight)
    if not math.isfinite(crps) or not math.isfinite(mase) or crps < 0.0 or mase < 0.0:
        raise ValueError("teacher utility weights must be finite and nonnegative.")
    total = crps + mase
    if total <= 0.0:
        raise ValueError("At least one teacher utility metric weight must be positive.")
    return float(crps / total), float(mase / total)


def _metric_weight_alias(target_key: str) -> str:
    key = str(target_key)
    if key.startswith("u_") and key.endswith("_uniform"):
        return key[2:-8]
    if key.startswith("u_") and key.endswith("_best"):
        return key[2:-5]
    return key


def validate_teacher_metric_target_keys(keys: Sequence[str] | str | None) -> Tuple[str, ...]:
    if keys is None:
        return TEACHER_METRIC_TARGET_KEYS
    if isinstance(keys, str):
        raw = [part.strip() for part in keys.split(",")]
    else:
        raw = [str(part).strip() for part in keys]
    out = tuple(part for part in raw if part)
    if not out:
        raise ValueError("teacher_metric_target_keys must contain at least one utility column.")
    duplicates = sorted({key for key in out if out.count(key) > 1})
    if duplicates:
        raise ValueError(f"teacher_metric_target_keys contains duplicates: {duplicates}")
    return out


def normalize_teacher_utility_weights(
    target_keys: Sequence[str],
    weights: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    keys = validate_teacher_metric_target_keys(target_keys)
    raw = dict(weights or {})
    default_weights = objective_weight_map_for_keys(keys)
    values: List[float] = []
    for key in keys:
        alias = _metric_weight_alias(key)
        if key in raw:
            value = float(raw[key])
        elif alias in raw:
            value = float(raw[alias])
        elif f"{key}_weight" in raw:
            value = float(raw[f"{key}_weight"])
        elif f"{alias}_weight" in raw:
            value = float(raw[f"{alias}_weight"])
        else:
            value = float(default_weights.get(key, 1.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("teacher utility weights must be finite and nonnegative.")
        values.append(value)
    total = float(sum(values))
    if total <= 0.0:
        raise ValueError("At least one teacher utility metric weight must be positive.")
    return {key: float(value / total) for key, value in zip(keys, values)}


def teacher_utility_weights_for_summary(target_keys: Sequence[str], weights: Mapping[str, float] | None) -> Dict[str, float]:
    normalized = normalize_teacher_utility_weights(target_keys, weights)
    if tuple(target_keys) == DEFAULT_TEACHER_METRIC_TARGET_KEYS:
        return {_metric_weight_alias(key): float(value) for key, value in normalized.items()}
    return {str(key): float(value) for key, value in normalized.items()}


def _finite_positive(value: Any, *, label: str) -> float:
    val = float(value)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"{label} must be finite and positive, got {value!r}.")
    return val


def validate_teacher_objective_hyperparameters(
    *,
    rank_temperature: float,
    regression_weight: float,
    pair_margin: float,
) -> Tuple[float, float, float]:
    temp = float(rank_temperature)
    reg = float(regression_weight)
    margin = float(pair_margin)
    if not math.isfinite(temp) or temp <= 0.0:
        raise ValueError(f"teacher_rank_temperature must be finite and positive, got {rank_temperature!r}.")
    if not math.isfinite(reg) or reg < 0.0:
        raise ValueError(f"teacher_regression_weight must be finite and nonnegative, got {regression_weight!r}.")
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError(f"teacher_pair_margin must be finite and nonnegative, got {pair_margin!r}.")
    return temp, reg, margin


def _summary_percentiles(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([float(value) for value in values], dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _teacher_candidate_weights(utilities: Sequence[float], *, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-6)
    logits = np.asarray(utilities, dtype=np.float64) / temp
    logits = logits - float(np.max(logits))
    weights = np.exp(logits)
    return weights / max(float(np.sum(weights)), 1e-12)


def _teacher_candidate_ess(weights: Sequence[float]) -> float:
    arr = np.asarray(weights, dtype=np.float64)
    return float(1.0 / max(float(np.sum(np.square(arr))), 1e-12))


def _resolve_setting_encoder_config(
    setting_feature_mode: str,
    setting_encoder_config: Mapping[str, Any] | SettingEncoderConfig | None,
) -> SettingEncoderConfig:
    if setting_encoder_config is None:
        return setting_encoder_config_from_payload({"mode": validate_setting_feature_mode(setting_feature_mode)})
    return setting_encoder_config_from_payload(setting_encoder_config)


def _setting_features_for_config(
    solver_key: str,
    target_nfe: int,
    *,
    setting_feature_mode: str,
    setting_encoder_config: Mapping[str, Any] | SettingEncoderConfig | None,
) -> torch.Tensor:
    config = _resolve_setting_encoder_config(setting_feature_mode, setting_encoder_config)
    return setting_features(str(solver_key), int(target_nfe), mode=setting_feature_mode, config=config)


def _json_hash(payload: Mapping[str, Any], *, prefix: str) -> str:
    return _canonical_json_hash(payload, prefix=prefix)


def stable_context_id(
    *,
    dataset: str,
    split_phase: str,
    example_idx: int,
    series_id: str,
    series_idx: int,
    target_t: int,
    history_start: int | None = None,
    history_stop: int | None = None,
    context_schema: str = "forecast_window",
) -> str:
    return _json_hash(
        {
            "context_schema": str(context_schema),
            "dataset": str(dataset),
            "split_phase": str(split_phase),
            "example_idx": int(example_idx),
            "series_id": str(series_id),
            "series_idx": int(series_idx),
            "target_t": int(target_t),
            "history_start": None if history_start is None else int(history_start),
            "history_stop": None if history_stop is None else int(history_stop),
        },
        prefix="ctx",
    )


def _optional_int(value: Any) -> int | None:
    if value is None or str(value) == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def context_id_from_row(row: MetricRow) -> str:
    existing = str(row.get("context_id", "") or "").strip()
    if existing:
        return existing
    dataset = str(row.get("dataset", row.get("dataset_key", ""))).strip()
    split_phase = str(row.get("split_phase", row.get("split", ""))).strip()
    example_idx_raw = row.get("example_idx", row.get("example_index", None))
    target_t_raw = row.get("target_t", None)
    has_series_identity = str(row.get("series_id", "")).strip() != "" or str(row.get("series_idx", "")).strip() != ""
    context_schema = str(row.get("context_schema", "") or "").strip()
    if context_schema and str(row.get("axis_dataset", "") or row.get("dataset", row.get("dataset_key", ""))).strip():
        payload = {
            "context_schema": context_schema,
            "dataset": str(row.get("axis_dataset", row.get("dataset", row.get("dataset_key", "")))),
            "split_phase": split_phase,
            "axis_series": str(row.get("axis_series", row.get("series_id", row.get("series_idx", "")))),
            "axis_time_bin": str(row.get("axis_time_bin", "")),
            "axis_record": str(row.get("axis_record", row.get("record_id", ""))),
            "axis_window": str(row.get("axis_window", "")),
            "axis_stratum": str(row.get("axis_stratum", row.get("stratum", ""))),
            "axis_member": str(row.get("axis_member", row.get("member_key", ""))),
            "axis_formula": str(row.get("axis_formula", row.get("formula", ""))),
            "axis_atom_count": str(row.get("axis_atom_count", row.get("atom_count", ""))),
            "axis_trajectory": str(row.get("axis_trajectory", row.get("trajectory_key", row.get("trajectory_id", "")))),
            "axis_iso_id": str(row.get("axis_iso_id", row.get("iso_id", ""))),
            "axis_flags": str(row.get("axis_flags", "")),
            "example_idx": str(row.get("example_idx", row.get("example_index", ""))),
            "target_t": str(row.get("target_t", "")),
            "history_start": str(row.get("history_start", "")),
            "history_stop": str(row.get("history_stop", "")),
        }
        return _json_hash(payload, prefix="ctx")
    missing = []
    if not dataset:
        missing.append("dataset")
    if not split_phase:
        missing.append("split_phase")
    if example_idx_raw is None or str(example_idx_raw) == "":
        missing.append("example_idx")
    if not has_series_identity:
        missing.append("series_id_or_series_idx")
    if target_t_raw is None or str(target_t_raw) == "":
        missing.append("target_t")
    if missing:
        raise ValueError(f"Context rows require context_id or complete identity fields; missing {missing}.")
    return stable_context_id(
        dataset=dataset,
        split_phase=split_phase,
        example_idx=int(example_idx_raw),
        series_id=str(row.get("series_id", "")),
        series_idx=int(row.get("series_idx", 0) or 0),
        target_t=int(target_t_raw),
        history_start=_optional_int(row.get("history_start")),
        history_stop=_optional_int(row.get("history_stop")),
        context_schema=context_schema or "forecast_window",
    )


def context_embedding_id_from_row(row: MetricRow) -> str:
    """Return the canonical key for looking up frozen context embeddings.

    `context_id` is the logical reward-pairing key. Generic multi-family
    artifacts may reuse the same logical context across checkpoint maturities,
    so the embedding lookup must prefer the checkpoint-scoped
    `context_embedding_id` when present.
    """

    existing = str(row.get("context_embedding_id", "") or "").strip()
    if existing:
        return existing
    return context_id_from_row(row)


def logical_seed_from_row(row: MetricRow) -> int | None:
    explicit = row.get("logical_seed", "")
    if explicit is not None and str(explicit).strip() != "":
        return int(explicit)
    parent = str(row.get("parent_row_signature", "") or "")
    parts = parent.split("|")
    if len(parts) >= 3:
        try:
            return int(parts[2])
        except (TypeError, ValueError):
            pass
    return _optional_int(row.get("seed"))


def evaluation_seed_from_row(row: MetricRow) -> int | None:
    explicit = row.get("evaluation_seed", "")
    if explicit is not None and str(explicit).strip() != "":
        return int(explicit)
    parent = str(row.get("parent_row_signature", "") or "")
    parts = parent.split("|")
    if len(parts) >= 3:
        try:
            return int(parts[2])
        except (TypeError, ValueError):
            pass
    return _optional_int(row.get("seed"))


def realized_nfe_from_row(row: MetricRow) -> int | None:
    explicit = row.get("realized_nfe", "")
    if explicit is not None and str(explicit).strip() != "":
        return int(explicit)
    actual = row.get("actual_nfe", "")
    if actual is not None and str(actual).strip() != "":
        return int(actual)
    runtime = row.get("runtime_nfe", "")
    if runtime is not None and str(runtime).strip() != "":
        return int(runtime)
    target = row.get("target_nfe", "")
    if target is not None and str(target).strip() != "":
        return int(target)
    return None


def series_key_from_row(row: MetricRow) -> str:
    series_id = str(row.get("series_id", "") or "").strip()
    if series_id:
        return series_id
    series_idx = str(row.get("series_idx", "") or "").strip()
    if series_idx:
        return f"series_idx:{series_idx}"
    axis_series = str(row.get("axis_series", "") or "").strip()
    if axis_series:
        return f"axis_series:{axis_series}"
    axis_member = str(row.get("axis_member", row.get("member_key", "")) or "").strip()
    axis_stratum = str(row.get("axis_stratum", row.get("stratum", "")) or "").strip()
    axis_trajectory = str(row.get("axis_trajectory", row.get("trajectory_key", row.get("trajectory_id", ""))) or "").strip()
    if axis_member or axis_stratum or axis_trajectory:
        return "axis:" + "|".join(part for part in (axis_member, axis_stratum, axis_trajectory) if part)
    raise ValueError("Rows require series_id, series_idx, or generalized axis series fields for series-disjoint diagnostics.")


def context_pair_key(row: MetricRow, *, pair_on_seed: bool = True) -> ContextPairKey:
    seed = logical_seed_from_row(row) if pair_on_seed else None
    return (
        str(row.get("dataset", row.get("dataset_key", ""))),
        str(row["solver_key"]),
        int(row["target_nfe"]),
        context_id_from_row(row),
        seed,
    )


def teacher_selection_candidate_group_key(row: MetricRow) -> Tuple[str, str, int, int | None, int | None, int | None]:
    return (
        context_id_from_row(row),
        str(row["solver_key"]),
        int(row["target_nfe"]),
        realized_nfe_from_row(row),
        logical_seed_from_row(row),
        evaluation_seed_from_row(row),
    )

def complete_candidate_group_payload(key: Tuple[str, str, int, int | None, int | None, int | None]) -> Dict[str, Any]:
    return {
        "context_id": str(key[0]),
        "solver_key": str(key[1]),
        "target_nfe": int(key[2]),
        "realized_nfe": None if key[3] is None else int(key[3]),
        "logical_seed": None if key[4] is None else int(key[4]),
        "evaluation_seed": None if key[5] is None else int(key[5]),
    }


def _nfe_sequence_key(row: MetricRow) -> Tuple[str, int | None, str, str, str]:
    return (
        str(row.get("dataset", row.get("dataset_key", ""))),
        logical_seed_from_row(row),
        str(row["solver_key"]),
        context_id_from_row(row),
        series_key_from_row(row),
    )


def _student_target_groups(rows: Sequence[MetricRow]) -> List[Tuple[ContextPairKey, List[MetricRow]]]:
    grouped: Dict[ContextPairKey, List[MetricRow]] = defaultdict(list)
    for row in rows:
        grouped[context_pair_key(row, pair_on_seed=True)].append(row)
    return [(key, group) for key, group in sorted(grouped.items(), key=lambda item: item[0])]


def student_nfe_sequence_pairs(rows: Sequence[MetricRow]) -> List[Tuple[int, int, float]]:
    """Return adjacent target rows plus their positive NFE distance."""
    sequence_items: Dict[Tuple[str, int | None, str, str, str], List[Tuple[int, int]]] = defaultdict(list)
    for index, (_, group) in enumerate(_student_target_groups(rows)):
        if not group:
            continue
        first = group[0]
        sequence_items[_nfe_sequence_key(first)].append((int(first["target_nfe"]), int(index)))
    pairs: List[Tuple[int, int, float]] = []
    for items in sequence_items.values():
        ordered = sorted(items, key=lambda item: (int(item[0]), int(item[1])))
        for left, right in zip(ordered, ordered[1:]):
            delta = max(1.0, float(int(right[0]) - int(left[0])))
            pairs.append((int(left[1]), int(right[1]), delta))
    return pairs


def student_nfe_sequence_pair_indices(rows: Sequence[MetricRow]) -> List[Tuple[int, int]]:
    """Return adjacent target row indices that share context/solver/series and differ only by ordered NFE."""
    return [(left, right) for left, right, _ in student_nfe_sequence_pairs(rows)]


def _physical_nfe_sequence_key(row: MetricRow) -> Tuple[Any, ...]:
    return (
        str(row.get("dataset", row.get("dataset_key", ""))),
        str(row.get("source_split_phase") or row.get("split_phase", row.get("split", ""))),
        _optional_int(row.get("seed")),
        str(row["solver_key"]),
        row.get("example_idx", row.get("example_index", "")),
        series_key_from_row(row),
        row.get("target_t", ""),
        _optional_int(row.get("history_start")),
        _optional_int(row.get("history_stop")),
    )


def nfe_sequence_diagnostic_summary(rows: Sequence[MetricRow]) -> Dict[str, Any]:
    sequence_groups: Dict[Tuple[str, int | None, str, str, str], set[int]] = defaultdict(set)
    physical_groups: Dict[Tuple[Any, ...], set[int]] = defaultdict(set)
    for row in rows:
        nfe = int(row["target_nfe"])
        sequence_groups[_nfe_sequence_key(row)].add(nfe)
        physical_groups[_physical_nfe_sequence_key(row)].add(nfe)
    sequence_nfe_sets: Dict[Tuple[int, ...], int] = defaultdict(int)
    physical_nfe_sets: Dict[Tuple[int, ...], int] = defaultdict(int)
    for nfes in sequence_groups.values():
        sequence_nfe_sets[tuple(sorted(nfes))] += 1
    for nfes in physical_groups.values():
        physical_nfe_sets[tuple(sorted(nfes))] += 1
    return {
        "row_count": int(len(rows)),
        "observed_target_nfes": sorted({int(row["target_nfe"]) for row in rows}),
        "split_phases": sorted({str(row.get("source_split_phase") or row.get("split_phase", row.get("split", ""))) for row in rows}),
        "nfe_sequence_pair_count": int(len(student_nfe_sequence_pairs(rows))),
        "sequence_group_count": int(len(sequence_groups)),
        "sequence_multi_nfe_group_count": int(sum(1 for nfes in sequence_groups.values() if len(nfes) > 1)),
        "sequence_nfe_sets_top": [
            {"nfes": list(nfes), "count": int(count)}
            for nfes, count in sorted(sequence_nfe_sets.items(), key=lambda item: (-item[1], item[0]))[:8]
        ],
        "physical_group_count": int(len(physical_groups)),
        "physical_multi_nfe_group_count": int(sum(1 for nfes in physical_groups.values() if len(nfes) > 1)),
        "physical_nfe_sets_top": [
            {"nfes": list(nfes), "count": int(count)}
            for nfes, count in sorted(physical_nfe_sets.items(), key=lambda item: (-item[1], item[0]))[:8]
        ],
    }



def validate_gipo_support_schedule_keys(
    support_schedule_keys: Sequence[str],
    *,
    allowed_schedule_keys: Sequence[str] = EXPERIMENTAL_SUPERVISION_SCHEDULE_KEYS,
) -> Tuple[str, ...]:
    keys = tuple(str(key) for key in support_schedule_keys)
    if not keys:
        raise ValueError("support_schedule_keys must not be empty.")
    bo_like = sorted(key for key in keys if "bo" in key.lower() or "candidate" in key.lower())
    if bo_like:
        raise ValueError(f"GIPO supervision must not include BO/candidate schedules: {bo_like}")
    allowed = {str(key) for key in allowed_schedule_keys}
    unsupported = sorted(set(keys) - allowed)
    if unsupported:
        raise ValueError(f"GIPO supervision is fixed/SER only; unsupported schedules: {unsupported}")
    return keys


def density_family_for_schedule_key(schedule_key: str) -> str:
    key = str(schedule_key).strip()
    if key == UNIFORM_SCHEDULE_KEY:
        return "uniform_anchor"
    if key == SER_PTG_SCHEDULE_KEY:
        return "ser_oracle"
    if key in AVERAGED_SCHEDULE_COMPONENTS:
        return f"avg_{AVERAGED_SCHEDULE_COMPONENTS[key][0]}"
    suffix = "_reversed"
    return key[: -len(suffix)] if key.endswith(suffix) else key


def validate_density_family_holdout_schedule_keys(
    holdout_schedule_keys: Sequence[str],
    *,
    support_schedule_keys: Sequence[str],
) -> Tuple[str, ...]:
    keys = tuple(str(key).strip() for key in holdout_schedule_keys if str(key).strip())
    if not keys:
        return ()
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"density-family holdout schedule keys must be unique; duplicates={duplicates}.")
    if UNIFORM_SCHEDULE_KEY in keys:
        raise ValueError("density-family holdout must not include uniform; uniform is the reward anchor.")
    support = set(validate_gipo_support_schedule_keys(support_schedule_keys))
    unsupported = sorted(set(keys) - support)
    if unsupported:
        raise ValueError(f"density-family holdout keys must be in support_schedule_keys; unsupported={unsupported}.")
    return keys


def split_rows_by_density_family_holdout(
    rows: Sequence[MetricRow],
    *,
    holdout_schedule_keys: Sequence[str],
    support_schedule_keys: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    keys = validate_density_family_holdout_schedule_keys(
        holdout_schedule_keys,
        support_schedule_keys=support_schedule_keys,
    )
    holdout_set = set(keys)
    if holdout_set:
        unrewarded_count = sum(
            1
            for row in rows
            if str(row.get("gipo_reward_protocol", "")).strip() != GIPO_PROTOCOL
            or "u_comp_uniform" not in row
        )
        if unrewarded_count:
            raise ValueError(
                "density-family holdout must run after uniform-anchored reward construction; "
                f"found {unrewarded_count} unrewarded rows."
            )
    fit_rows: List[Dict[str, Any]] = []
    holdout_rows: List[Dict[str, Any]] = []
    missing_rewards = [
        str(row["scheduler_key"])
        for row in rows
        if str(row["scheduler_key"]) in holdout_set
        and ("gipo_reward_protocol" not in row or "u_comp_uniform" not in row)
    ]
    if missing_rewards:
        raise ValueError(
            "Density-family holdout must run after uniform-anchored reward construction; "
            f"missing reward columns for schedules {sorted(set(missing_rewards))}."
        )
    for row in rows:
        copied = dict(row)
        schedule_key = str(copied["scheduler_key"])
        copied["density_family"] = density_family_for_schedule_key(schedule_key)
        if schedule_key in holdout_set:
            holdout_rows.append(copied)
        else:
            fit_rows.append(copied)
    metadata = {
        "density_family_holdout_schedule_keys": list(keys),
        "density_family_holdout_families": sorted({density_family_for_schedule_key(key) for key in keys}),
        "density_family_holdout_row_count": int(len(holdout_rows)),
        "density_family_fit_row_count": int(len(fit_rows)),
        "reward_anchor_schedule_key": UNIFORM_SCHEDULE_KEY,
    }
    return fit_rows, holdout_rows, metadata


def attach_uniform_gipo_rewards(
    rows: Iterable[MetricRow],
    *,
    support_schedule_keys: Sequence[str] | None = None,
    uniform_schedule_key: str = UNIFORM_SCHEDULE_KEY,
    utility_crps_weight: float = 0.5,
    utility_mase_weight: float = 0.5,
    pair_on_seed: bool = True,
    eps: float = DEFAULT_REWARD_EPS,
) -> List[Dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    crps_weight, mase_weight = _normalized_metric_weights(utility_crps_weight, utility_mase_weight)
    support_keys = validate_gipo_support_schedule_keys(
        DEFAULT_SUPERVISION_SCHEDULE_KEYS if support_schedule_keys is None else support_schedule_keys
    )
    support = {str(key) for key in support_keys}
    uniform_key = str(uniform_schedule_key)
    grouped: Dict[ContextPairKey, List[Dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        schedule_key = str(row["scheduler_key"])
        if schedule_key not in support:
            raise ValueError(f"Unsupported GIPO supervision row {schedule_key!r}; expected fixed/SER support.")
        row["context_id"] = context_id_from_row(row)
        grouped[context_pair_key(row, pair_on_seed=pair_on_seed)].append(row)

    out: List[Dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: item[0]):
        counts = {schedule: 0 for schedule in support}
        for row in group:
            counts[str(row["scheduler_key"])] = counts.get(str(row["scheduler_key"]), 0) + 1
        bad_counts = {schedule: int(counts.get(schedule, 0)) for schedule in sorted(support) if int(counts.get(schedule, 0)) != 1}
        if bad_counts:
            raise ValueError(
                "Uniform-anchored context rewards require exactly one row for every support schedule "
                f"in paired context {key}; counts={bad_counts}."
            )
        uniform_rows = [row for row in group if str(row["scheduler_key"]) == uniform_key]
        if len(uniform_rows) != 1:
            raise ValueError(f"Uniform-anchored context rewards require exactly one uniform row in paired context {key}.")
        for row in group:
            reward_columns = uniform_anchored_objective_columns(
                row,
                uniform_rows[0],
                FORECAST_METRIC_SPECS,
                uniform_schedule_key=uniform_key,
                eps=float(eps),
            )
            u_crps = float(reward_columns["u_crps_uniform"])
            u_mase = float(reward_columns["u_mase_uniform"])
            copied = dict(row)
            copied.update(
                {
                    "gipo_reward_protocol": GIPO_PROTOCOL,
                    "reward_anchor_schedule_key": uniform_key,
                    **reward_columns,
                    "u_comp_crps_weight": float(crps_weight),
                    "u_comp_mase_weight": float(mase_weight),
                    "reward_metric_weights_json": json.dumps(
                        {
                            "u_crps_uniform": float(crps_weight),
                            "u_mase_uniform": float(mase_weight),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "u_comp_uniform": float(crps_weight * u_crps + mase_weight * u_mase),
                }
            )
            out.append(copied)
    return out


def split_rows_by_context_holdout(
    rows: Sequence[MetricRow],
    *,
    holdout_fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    frac = float(holdout_fraction)
    if frac <= 0.0:
        return [dict(row) for row in rows], []
    if frac >= 1.0:
        raise ValueError("holdout_fraction must be smaller than 1.0.")
    context_ids = sorted({context_id_from_row(row) for row in rows})
    if not context_ids:
        return [], []
    rng = np.random.default_rng(int(seed))
    shuffled = list(context_ids)
    rng.shuffle(shuffled)
    holdout_count = max(1, int(round(float(len(shuffled)) * frac)))
    holdout_ids = set(shuffled[:holdout_count])
    fit_rows: List[Dict[str, Any]] = []
    holdout_rows: List[Dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["context_id"] = context_id_from_row(copied)
        if copied["context_id"] in holdout_ids:
            holdout_rows.append(copied)
        else:
            fit_rows.append(copied)
    return fit_rows, holdout_rows


def recommended_context_calibration_count(
    available_contexts: int,
    *,
    normalized_combined_reference: int | None = None,
    cells: int = 12,
    default_total: int = DEFAULT_CONTEXT_CALIBRATION_TOTAL,
    min_total: int = MIN_CONTEXT_CALIBRATION_TOTAL,
    max_total: int = MAX_CONTEXT_CALIBRATION_TOTAL,
) -> int:
    available = int(available_contexts)
    if available <= 0:
        raise ValueError("available_contexts must be positive.")
    cap = int(max_total)
    requested = int(default_total) if normalized_combined_reference is None else int(round(0.20 * float(normalized_combined_reference)))
    target = min(cap, max(int(min_total), requested))
    return int(min(available, target))


def context_calibration_train_val_counts(
    available_contexts: int,
    *,
    validation_available: int | None = None,
    normalized_combined_reference: int | None = None,
    validation_fraction: float = DEFAULT_CONTEXT_CALIBRATION_VALIDATION_FRACTION,
) -> Tuple[int, int]:
    total = recommended_context_calibration_count(
        int(available_contexts),
        normalized_combined_reference=normalized_combined_reference,
    )
    val_cap = int(available_contexts if validation_available is None else validation_available)
    val_target = max(18, int(round(float(total) * float(validation_fraction))))
    val_count = int(min(val_cap, max(0, min(total - 1, val_target))))
    train_count = int(total - val_count)
    if train_count <= 0:
        raise ValueError("Context calibration split must leave at least one train context.")
    return train_count, val_count


def sample_context_ids_stratified(
    rows: Sequence[MetricRow],
    *,
    sample_count: int | None = None,
    seed: int = 0,
) -> List[str]:
    by_context: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        context_id = context_id_from_row(row)
        if context_id not in by_context:
            by_context[context_id] = dict(row)
    if not by_context:
        return []
    target = int(sample_count) if sample_count is not None else recommended_context_calibration_count(len(by_context))
    target = int(min(len(by_context), max(1, target)))
    if target >= len(by_context):
        return sorted(by_context)
    strata: Dict[Tuple[str, str, str, str, str, str], List[str]] = defaultdict(list)
    for context_id, row in by_context.items():
        dataset = str(row.get("axis_dataset", row.get("dataset", row.get("dataset_key", ""))))
        series = str(row.get("axis_series", row.get("series_id", row.get("series_idx", ""))))
        stratum = str(row.get("axis_stratum", row.get("stratum", "")))
        formula = str(row.get("axis_formula", row.get("formula", "")))
        trajectory = str(row.get("axis_trajectory", row.get("trajectory_key", row.get("trajectory_id", ""))))
        explicit_time_bin = str(row.get("axis_time_bin", "") or "")
        try:
            target_t = int(row.get("target_t", 0))
        except (TypeError, ValueError):
            target_t = 0
        time_bin = explicit_time_bin or str(int(math.floor(float(target_t) / 24.0)))
        strata[(dataset, series, stratum, formula, trajectory, time_bin)].append(context_id)
    rng = np.random.default_rng(int(seed))
    selected: List[str] = []
    stratum_items = sorted(strata.items(), key=lambda item: item[0])
    allocations: List[Tuple[float, int, int]] = []
    for idx, (_, ids) in enumerate(stratum_items):
        raw = float(target) * float(len(ids)) / float(len(by_context))
        base = min(len(ids), int(math.floor(raw)))
        allocations.append((raw - base, idx, base))
    counts = [base for _, _, base in allocations]
    remaining = int(target - sum(counts))
    for _, idx, _ in sorted(allocations, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        if counts[idx] < len(stratum_items[idx][1]):
            counts[idx] += 1
            remaining -= 1
    for (_, ids), keep in zip(stratum_items, counts):
        if keep <= 0:
            continue
        local = list(ids)
        rng.shuffle(local)
        selected.extend(local[:keep])
    if len(selected) < target:
        missing = sorted(set(by_context) - set(selected))
        rng.shuffle(missing)
        selected.extend(missing[: int(target - len(selected))])
    return sorted(set(selected))[:target]


@dataclass(frozen=True)
class EmbeddingNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, embeddings: Mapping[str, Sequence[float]], context_ids: Iterable[str]) -> "EmbeddingNormalizer":
        ids = [str(context_id) for context_id in context_ids]
        if not ids:
            raise ValueError("Embedding normalization requires at least one train context.")
        missing = sorted(context_id for context_id in ids if context_id not in embeddings)
        if missing:
            raise KeyError(f"Missing context embeddings for train contexts: {missing[:8]}")
        matrix = np.asarray([embeddings[context_id] for context_id in ids], dtype=np.float32)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return cls(mean=mean.astype(np.float32), std=std)

    def transform_one(self, embedding: Sequence[float]) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.shape != self.mean.shape:
            raise ValueError(f"Embedding shape {vector.shape} does not match normalizer shape {self.mean.shape}.")
        return ((vector - self.mean) / self.std).astype(np.float32)

    def transform_table(self, embeddings: Mapping[str, Sequence[float]]) -> Dict[str, np.ndarray]:
        return {str(context_id): self.transform_one(vector) for context_id, vector in embeddings.items()}

    def to_payload(self) -> Dict[str, Any]:
        return {"mean": self.mean.astype(float).tolist(), "std": self.std.astype(float).tolist()}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EmbeddingNormalizer":
        return cls(mean=np.asarray(payload["mean"], dtype=np.float32), std=np.asarray(payload["std"], dtype=np.float32))


@dataclass(frozen=True)
class DensityFeatureNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, density_masses: Iterable[Sequence[float]], *, reference_time_grid: Sequence[float]) -> "DensityFeatureNormalizer":
        features = [density_log_features(mass, reference_time_grid=reference_time_grid) for mass in density_masses]
        if not features:
            raise ValueError("Density feature normalization requires at least one density mass.")
        matrix = np.asarray(features, dtype=np.float32)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return cls(mean=mean.astype(np.float32), std=std)

    def transform_one(self, density_mass: Sequence[float], *, reference_time_grid: Sequence[float]) -> np.ndarray:
        features = density_log_features(density_mass, reference_time_grid=reference_time_grid)
        if features.shape != self.mean.shape:
            raise ValueError(f"Density feature shape {features.shape} does not match normalizer shape {self.mean.shape}.")
        return ((features - self.mean) / self.std).astype(np.float32)

    def to_payload(self) -> Dict[str, Any]:
        return {"mean": self.mean.astype(float).tolist(), "std": self.std.astype(float).tolist()}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DensityFeatureNormalizer":
        return cls(mean=np.asarray(payload["mean"], dtype=np.float32), std=np.asarray(payload["std"], dtype=np.float32))


def _sanitize_public_manifest_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize_public_manifest_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_public_manifest_value(item) for item in value]
    if isinstance(value, str):
        text = str(value)
        looks_absolute = Path(text).is_absolute() or text.startswith("/") or (
            len(text) >= 3 and text[1:3] == ":\\"
        )
        if looks_absolute:
            return Path(text).name
    return value


def save_context_embedding_table(
    path: str | Path,
    embeddings: Mapping[str, Sequence[float]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    sorted_ids = sorted(str(key) for key in embeddings)
    if not sorted_ids:
        raise ValueError("Cannot save an empty context embedding table.")
    max_id_len = max(1, max(len(context_id) for context_id in sorted_ids))
    context_ids = np.asarray(sorted_ids, dtype=f"<U{max_id_len}")
    matrix = np.asarray([embeddings[str(context_id)] for context_id in context_ids.tolist()], dtype=np.float32)
    np.savez_compressed(resolved, context_ids=context_ids, embeddings=matrix)
    manifest = {
        "artifact": "context_embedding_table",
        "protocol": GIPO_PROTOCOL,
        "path": resolved.name,
        "context_count": int(context_ids.size),
        "embedding_dim": int(matrix.shape[1]),
        "metadata": _sanitize_public_manifest_value(dict(metadata or {})),
    }
    manifest_path = resolved.with_suffix(resolved.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_context_embedding_table(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as payload:
        context_ids = [str(value) for value in payload["context_ids"].tolist()]
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(context_ids):
        raise ValueError("Context embedding table has inconsistent ids and matrix shape.")
    return {context_id: embeddings[idx].astype(np.float32, copy=True) for idx, context_id in enumerate(context_ids)}


def build_series_index_map(rows: Iterable[MetricRow]) -> Dict[str, int]:
    keys = sorted({series_key_from_row(row) for row in rows})
    return {key: idx for idx, key in enumerate(keys)}


def remapped_series_index(row: MetricRow, series_index_map: Mapping[str, int]) -> int:
    key = series_key_from_row(row)
    if key not in series_index_map:
        return len(series_index_map)
    return int(series_index_map[key])


def _student_target_representative_rows(rows: Sequence[MetricRow]) -> List[MetricRow]:
    return [group[0] for _, group in _student_target_groups(rows) if group]


def grid_for_schedule(
    schedule_key: str,
    solver_key: str,
    target_nfe: int,
    *,
    schedule_grids: Mapping[ScheduleGridKey, Sequence[float]] | None = None,
    checkpoint_step: int | None = None,
) -> Tuple[float, ...]:
    key = str(schedule_key)
    macro_steps = solver_macro_steps(str(solver_key), int(target_nfe))
    if key in AVERAGED_SCHEDULE_COMPONENTS:
        left_key, right_key = AVERAGED_SCHEDULE_COMPONENTS[key]
        left_grid = grid_for_schedule(
            left_key,
            solver_key,
            target_nfe,
            schedule_grids=schedule_grids,
            checkpoint_step=checkpoint_step,
        )
        right_grid = grid_for_schedule(
            right_key,
            solver_key,
            target_nfe,
            schedule_grids=schedule_grids,
            checkpoint_step=checkpoint_step,
        )
        reference = uniform_reference_grid()
        left_mass = grid_to_density_mass(left_grid, reference_time_grid=reference, macro_steps=macro_steps)
        right_mass = grid_to_density_mass(right_grid, reference_time_grid=reference, macro_steps=macro_steps)
        avg_mass = average_density_masses(left_mass, right_mass)
        return density_mass_to_time_grid(
            avg_mass,
            reference_time_grid=reference,
            macro_steps=macro_steps,
        )
    if key in REVERSED_SCHEDULE_BASE and key not in EXPERIMENTAL_FIXED_SCHEDULE_KEYS:
        base_grid = grid_for_schedule(
            REVERSED_SCHEDULE_BASE[key],
            solver_key,
            target_nfe,
            schedule_grids=schedule_grids,
            checkpoint_step=checkpoint_step,
        )
        reversed_grid = [1.0 - float(value) for value in reversed(tuple(base_grid))]
        return validate_time_grid(reversed_grid, macro_steps=macro_steps)
    if schedule_grids is not None:
        base_key = (key, str(solver_key), int(target_nfe))
        if checkpoint_step is not None:
            checkpoint_key = (*base_key, int(checkpoint_step))
            if checkpoint_key in schedule_grids:
                return validate_time_grid(schedule_grids[checkpoint_key], macro_steps=macro_steps)
        if base_key in schedule_grids:
            return validate_time_grid(schedule_grids[base_key], macro_steps=macro_steps)
    if key in EXPERIMENTAL_FIXED_SCHEDULE_KEYS:
        grid = build_schedule_grid(key, macro_steps)
        if grid is None:
            raise ValueError(f"No fixed schedule grid for {key}.")
        return validate_time_grid(grid, macro_steps=macro_steps)
    raise KeyError(f"Missing schedule grid for {(key, str(solver_key), int(target_nfe))}.")


def density_mass_for_row(
    row: MetricRow,
    *,
    schedule_grids: Mapping[ScheduleGridKey, Sequence[float]] | None,
    reference_time_grid: Sequence[float],
) -> Tuple[float, ...]:
    solver = str(row["solver_key"])
    target_nfe = int(row["target_nfe"])
    checkpoint_step = None
    if row.get("checkpoint_step", "") not in (None, ""):
        checkpoint_step = int(row["checkpoint_step"])
    grid = grid_for_schedule(
        str(row["scheduler_key"]),
        solver,
        target_nfe,
        schedule_grids=schedule_grids,
        checkpoint_step=checkpoint_step,
    )
    return grid_to_density_mass(grid, reference_time_grid=reference_time_grid, macro_steps=solver_macro_steps(solver, target_nfe))


def _density_bin_geometry(density_dim: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    bins = int(density_dim)
    if bins <= 0:
        raise ValueError("density_dim must be positive.")
    edges = torch.linspace(0.0, 1.0, bins + 1, dtype=dtype, device=device)
    left = edges[:-1]
    right = edges[1:]
    centers = 0.5 * (left + right)
    widths = torch.clamp(right - left, min=1e-12)
    return torch.stack([centers, widths], dim=-1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    even_width = int(x.shape[-1] // 2 * 2)
    if even_width == 0:
        return x
    core = x[..., :even_width]
    rest = x[..., even_width:]
    first = core[..., 0::2]
    second = core[..., 1::2]
    rotated = torch.stack((-second, first), dim=-1).reshape_as(core)
    return torch.cat([rotated, rest], dim=-1)


def _apply_density_bin_rope(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError("Density RoPE expects [batch, bins, heads, head_dim] q/k tensors.")
    bins = int(x.shape[1])
    head_dim = int(x.shape[-1])
    even_width = int(head_dim // 2 * 2)
    if bins <= 1 or even_width == 0:
        return x
    centers = (torch.arange(bins, dtype=x.dtype, device=x.device) + 0.5) / float(bins)
    frequencies = torch.arange(1, even_width // 2 + 1, dtype=x.dtype, device=x.device)
    angles = centers[:, None] * frequencies[None, :] * math.pi
    sin = torch.repeat_interleave(torch.sin(angles), repeats=2, dim=-1)[None, :, None, :]
    cos = torch.repeat_interleave(torch.cos(angles), repeats=2, dim=-1)[None, :, None, :]
    core = x[..., :even_width]
    rest = x[..., even_width:]
    rotated = core * cos + _rotate_half(core) * sin
    return torch.cat([rotated, rest], dim=-1)


class _DensityTokenRoPESelfAttention(nn.Module):
    def __init__(self, *, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        hidden = int(hidden_dim)
        self.hidden_dim = hidden
        self.heads = int(heads)
        self.head_dim = hidden // self.heads
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out = nn.Linear(hidden, hidden)
        self.dropout_probability = float(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("Density token RoPE attention expects a [batch, bins, hidden] tensor.")
        batch, bins, _ = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, bins, 3, self.heads, self.head_dim)
        q = _apply_density_bin_rope(qkv[:, :, 0]).permute(0, 2, 1, 3)
        k = _apply_density_bin_rope(qkv[:, :, 1]).permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_probability if self.training else 0.0,
        )
        return self.out(attended.permute(0, 2, 1, 3).reshape(batch, bins, self.hidden_dim))


class _DensityTokenTransformerBlock(nn.Module):
    def __init__(self, *, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        if int(hidden_dim) % int(heads) != 0:
            raise ValueError("Transformer hidden_dim must be divisible by attention heads.")
        hidden = int(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = _DensityTokenRoPESelfAttention(hidden_dim=hidden, heads=int(heads), dropout=float(dropout))
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden * 4, hidden),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("Density token attention expects a [batch, bins, hidden] tensor.")
        tokens = tokens + self.dropout(self.attn(self.norm1(tokens)))
        tokens = tokens + self.dropout(self.ff(self.norm2(tokens)))
        return tokens


class _DensityConditioningMixin:
    def _condition_embedding(
        self,
        setting_feature_batch: torch.Tensor,
        series_index_batch: torch.Tensor,
        context_embedding_batch: torch.Tensor,
        *,
        rows: Sequence[MetricRow] | None,
    ) -> torch.Tensor:
        batch = int(setting_feature_batch.shape[0])
        if setting_feature_batch.ndim != 2 or setting_feature_batch.shape[-1] != self.setting_dim:
            raise ValueError("setting_feature_batch must be 2D with setting_dim columns.")
        if context_embedding_batch.ndim != 2 or context_embedding_batch.shape[-1] != self.context_dim:
            raise ValueError("context_embedding_batch must be 2D with context_dim columns.")
        if context_embedding_batch.shape[0] != batch:
            raise ValueError("GIPO conditioning batches must share the same batch dimension.")
        z = torch.cat(
            [
                setting_feature_batch,
                context_embedding_batch.to(device=setting_feature_batch.device, dtype=setting_feature_batch.dtype),
            ],
            dim=-1,
        )
        return self.condition_mlp(z)

    def _encode_density_tokens(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        out = tokens
        if condition.ndim != 2 or condition.shape[0] != tokens.shape[0] or condition.shape[-1] != tokens.shape[-1]:
            raise ValueError("GIPO conditioning expects a [batch, hidden] tensor aligned with density tokens.")
        out = out + condition.unsqueeze(1)
        for block in self.blocks:
            out = block(out)
        return self.final_norm(out)

    def model_config(self) -> Dict[str, Any]:
        config = {
            "architecture": self.architecture,
            "setting_dim": int(self.setting_dim),
            "density_dim": int(self.density_dim),
            "context_dim": int(self.context_dim),
            "num_series": int(self.num_series),
            "series_feature_dim": int(self.series_feature_dim),
            "series_conditioning": SERIES_CONDITIONING_NONE_CONTEXT_ONLY,
            "hidden_dim": int(self.hidden_dim),
            "hidden_layers": int(self.hidden_layers),
            "attention_heads": int(self.attention_heads),
            "dropout": float(self.dropout_probability),
            "conditioning_style": self.conditioning_style,
            "density_token_attention": DENSITY_TOKEN_ATTENTION_ROPE,
            "density_feature_mean": self.density_feature_mean.detach().cpu().numpy().astype(float).tolist()
            if hasattr(self, "density_feature_mean")
            else None,
            "density_feature_std": self.density_feature_std.detach().cpu().numpy().astype(float).tolist()
            if hasattr(self, "density_feature_std")
            else None,
        }
        if self.architecture == ARCHITECTURE_DENSITY_FORM_TRANSFORMER:
            config.update(
                {
                    "teacher_output": TEACHER_OUTPUT_METRIC_VECTOR,
                    "teacher_metric_targets": list(self.teacher_metric_targets),
                    "teacher_metric_target_protocol": TEACHER_METRIC_TARGET_PROTOCOL_VECTOR,
                    "teacher_metric_mask_protocol": TEACHER_METRIC_MASK_PROTOCOL,
                    "teacher_scalarization": TEACHER_SCALARIZATION_WEIGHTED_AVERAGE,
                }
            )
        return config


class GIPODensityFormTeacherTransformer(_DensityConditioningMixin, nn.Module):
    """Teacher that scores an observed density form after bin-token self-attention."""

    architecture = ARCHITECTURE_DENSITY_FORM_TRANSFORMER

    def __init__(
        self,
        *,
        setting_dim: int,
        density_dim: int,
        context_dim: int,
        num_series: int,
        series_feature_dim: int = SERIES_CONDITIONING_DIM,
        hidden_dim: int = DEFAULT_TRANSFORMER_HIDDEN_DIM,
        hidden_layers: int = DEFAULT_TRANSFORMER_LAYERS,
        attention_heads: int = DEFAULT_TRANSFORMER_HEADS,
        dropout: float = DEFAULT_TRANSFORMER_DROPOUT,
        conditioning_style: str = CONDITIONING_STYLE_ADDITIVE_MLP,
        density_feature_mean: Sequence[float] | None = None,
        density_feature_std: Sequence[float] | None = None,
        teacher_metric_targets: Sequence[str] = TEACHER_METRIC_TARGET_KEYS,
    ):
        super().__init__()
        self.setting_dim = int(setting_dim)
        self.density_dim = int(density_dim)
        self.context_dim = int(context_dim)
        self.num_series = 0
        self.unknown_series_index = 0
        self.series_feature_dim = SERIES_CONDITIONING_DIM
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.attention_heads = validate_gipo_attention_heads(int(attention_heads))
        self.dropout_probability = float(dropout)
        self.conditioning_style = validate_canonical_conditioning_style(
            {"conditioning_style": conditioning_style},
        )
        self.teacher_metric_targets = validate_teacher_metric_target_keys(teacher_metric_targets)
        mean = np.zeros(int(density_dim), dtype=np.float32) if density_feature_mean is None else np.asarray(density_feature_mean, dtype=np.float32)
        std = np.ones(int(density_dim), dtype=np.float32) if density_feature_std is None else np.asarray(density_feature_std, dtype=np.float32)
        if mean.shape != (int(density_dim),) or std.shape != (int(density_dim),):
            raise ValueError("density_feature_mean/std must match density_dim.")
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        self.register_buffer("density_feature_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("density_feature_std", torch.tensor(std, dtype=torch.float32))
        condition_dim = int(setting_dim) + int(context_dim)
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.density_token_proj = nn.Sequential(nn.Linear(4, int(hidden_dim)), nn.SiLU(), nn.Linear(int(hidden_dim), int(hidden_dim)))
        self.bin_geometry_proj = nn.Linear(2, int(hidden_dim))
        self.blocks = nn.ModuleList(
            [
                _DensityTokenTransformerBlock(
                    hidden_dim=int(hidden_dim),
                    heads=int(self.attention_heads),
                    dropout=float(dropout),
                )
                for _ in range(int(hidden_layers))
            ]
        )
        self.final_norm = nn.LayerNorm(int(hidden_dim))
        self.head = nn.Sequential(nn.LayerNorm(int(hidden_dim)), nn.Linear(int(hidden_dim), len(self.teacher_metric_targets)))

    def forward(
        self,
        setting_feature_batch: torch.Tensor,
        density_feature_batch: torch.Tensor,
        series_index_batch: torch.Tensor,
        context_embedding_batch: torch.Tensor,
        *,
        rows: Sequence[MetricRow] | None = None,
    ) -> torch.Tensor:
        batch = int(setting_feature_batch.shape[0])
        if density_feature_batch.ndim != 2 or density_feature_batch.shape[-1] != self.density_dim:
            raise ValueError("density_feature_batch must be 2D with density_dim columns.")
        if density_feature_batch.shape[0] != batch:
            raise ValueError("Teacher feature batches must share the same batch dimension.")
        z = self._condition_embedding(
            setting_feature_batch,
            series_index_batch,
            context_embedding_batch,
            rows=rows,
        )
        normalized_log_density = density_feature_batch.to(device=setting_feature_batch.device, dtype=setting_feature_batch.dtype)
        log_density = normalized_log_density * self.density_feature_std.to(
            device=normalized_log_density.device,
            dtype=normalized_log_density.dtype,
        ) + self.density_feature_mean.to(device=normalized_log_density.device, dtype=normalized_log_density.dtype)
        geometry = _density_bin_geometry(
            self.density_dim,
            dtype=setting_feature_batch.dtype,
            device=setting_feature_batch.device,
        )
        token_geometry = geometry.unsqueeze(0).expand(batch, -1, -1)
        token_features = torch.cat([log_density.unsqueeze(-1), normalized_log_density.unsqueeze(-1), token_geometry], dim=-1)
        tokens = self.density_token_proj(token_features) + self.bin_geometry_proj(token_geometry)
        encoded = self._encode_density_tokens(tokens, z)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


class GIPODensityQueryStudentTransformer(_DensityConditioningMixin, nn.Module):
    """Student that decodes one density logit per bin query under conditioning z."""

    architecture = ARCHITECTURE_DENSITY_QUERY_TRANSFORMER

    def __init__(
        self,
        *,
        setting_dim: int,
        density_dim: int,
        context_dim: int,
        num_series: int,
        series_feature_dim: int = SERIES_CONDITIONING_DIM,
        hidden_dim: int = DEFAULT_TRANSFORMER_HIDDEN_DIM,
        hidden_layers: int = DEFAULT_TRANSFORMER_LAYERS,
        attention_heads: int = DEFAULT_TRANSFORMER_HEADS,
        dropout: float = DEFAULT_TRANSFORMER_DROPOUT,
        conditioning_style: str = CONDITIONING_STYLE_ADDITIVE_MLP,
    ):
        super().__init__()
        self.setting_dim = int(setting_dim)
        self.density_dim = int(density_dim)
        self.context_dim = int(context_dim)
        self.num_series = 0
        self.unknown_series_index = 0
        self.series_feature_dim = SERIES_CONDITIONING_DIM
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.attention_heads = validate_gipo_attention_heads(int(attention_heads))
        self.dropout_probability = float(dropout)
        self.conditioning_style = validate_canonical_conditioning_style(
            {"conditioning_style": conditioning_style},
        )
        condition_dim = int(setting_dim) + int(context_dim)
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.query_proj = nn.Sequential(nn.Linear(2, int(hidden_dim)), nn.SiLU(), nn.Linear(int(hidden_dim), int(hidden_dim)))
        self.blocks = nn.ModuleList(
            [
                _DensityTokenTransformerBlock(
                    hidden_dim=int(hidden_dim),
                    heads=int(self.attention_heads),
                    dropout=float(dropout),
                )
                for _ in range(int(hidden_layers))
            ]
        )
        self.final_norm = nn.LayerNorm(int(hidden_dim))
        self.head = nn.Linear(int(hidden_dim), 1)

    def logits(
        self,
        setting_feature_batch: torch.Tensor,
        series_index_batch: torch.Tensor,
        context_embedding_batch: torch.Tensor,
        *,
        rows: Sequence[MetricRow] | None = None,
    ) -> torch.Tensor:
        batch = int(setting_feature_batch.shape[0])
        z = self._condition_embedding(
            setting_feature_batch,
            series_index_batch,
            context_embedding_batch,
            rows=rows,
        )
        geometry = _density_bin_geometry(
            self.density_dim,
            dtype=setting_feature_batch.dtype,
            device=setting_feature_batch.device,
        )
        token_geometry = geometry.unsqueeze(0).expand(batch, -1, -1)
        tokens = self.query_proj(token_geometry)
        encoded = self._encode_density_tokens(tokens, z)
        return self.head(encoded).squeeze(-1)

    def density_mass(
        self,
        setting_feature_batch: torch.Tensor,
        series_index_batch: torch.Tensor,
        context_embedding_batch: torch.Tensor,
        *,
        rows: Sequence[MetricRow] | None = None,
    ) -> torch.Tensor:
        return torch.softmax(
            self.logits(
                setting_feature_batch,
                series_index_batch,
                context_embedding_batch,
                rows=rows,
            ),
            dim=-1,
        )


def build_gipo_teacher_model(
    *,
    architecture: str = ARCHITECTURE_DENSITY_FORM_TRANSFORMER,
    setting_dim: int,
    density_dim: int,
    context_dim: int,
    num_series: int,
    model_config: Mapping[str, Any] | None = None,
) -> nn.Module:
    validate_gipo_architecture(architecture, role="teacher")
    cfg = dict(model_config or {})
    conditioning_style = validate_canonical_conditioning_style(
        cfg,
    )
    validate_series_conditioning(cfg.get("series_conditioning"))
    validate_gipo_density_token_attention(cfg)
    validate_gipo_teacher_output(cfg, require_present=False)
    return GIPODensityFormTeacherTransformer(
        setting_dim=int(setting_dim),
        density_dim=int(density_dim),
        context_dim=int(context_dim),
        num_series=int(num_series),
        series_feature_dim=SERIES_CONDITIONING_DIM,
        hidden_dim=int(cfg.get("hidden_dim", DEFAULT_TRANSFORMER_HIDDEN_DIM)),
        hidden_layers=int(cfg.get("hidden_layers", cfg.get("transformer_layers", DEFAULT_TRANSFORMER_LAYERS))),
        attention_heads=validate_gipo_attention_heads(int(cfg.get("attention_heads", DEFAULT_TRANSFORMER_HEADS))),
        dropout=float(cfg.get("dropout", DEFAULT_TRANSFORMER_DROPOUT)),
        conditioning_style=conditioning_style,
        density_feature_mean=cfg.get("density_feature_mean"),
        density_feature_std=cfg.get("density_feature_std"),
        teacher_metric_targets=validate_teacher_metric_target_keys(cfg.get("teacher_metric_targets", TEACHER_METRIC_TARGET_KEYS)),
    )


def build_gipo_student_model(
    *,
    architecture: str = ARCHITECTURE_DENSITY_QUERY_TRANSFORMER,
    setting_dim: int,
    density_dim: int,
    context_dim: int,
    num_series: int,
    model_config: Mapping[str, Any] | None = None,
) -> nn.Module:
    validate_gipo_architecture(architecture, role="student")
    cfg = dict(model_config or {})
    conditioning_style = validate_canonical_conditioning_style(
        cfg,
    )
    validate_series_conditioning(cfg.get("series_conditioning"))
    validate_gipo_density_token_attention(cfg)
    return GIPODensityQueryStudentTransformer(
        setting_dim=int(setting_dim),
        density_dim=int(density_dim),
        context_dim=int(context_dim),
        num_series=int(num_series),
        series_feature_dim=SERIES_CONDITIONING_DIM,
        hidden_dim=int(cfg.get("hidden_dim", DEFAULT_TRANSFORMER_HIDDEN_DIM)),
        hidden_layers=int(cfg.get("hidden_layers", cfg.get("transformer_layers", DEFAULT_TRANSFORMER_LAYERS))),
        attention_heads=validate_gipo_attention_heads(int(cfg.get("attention_heads", DEFAULT_TRANSFORMER_HEADS))),
        dropout=float(cfg.get("dropout", DEFAULT_TRANSFORMER_DROPOUT)),
        conditioning_style=conditioning_style,
    )


def model_config_from_model(model: nn.Module) -> Dict[str, Any]:
    if hasattr(model, "model_config"):
        return dict(model.model_config())  # type: ignore[attr-defined]
    raise ValueError("GIPO model is missing canonical architecture metadata.")


def _teacher_architecture(model: nn.Module) -> str:
    return validate_gipo_architecture(str(getattr(model, "architecture", "")), role="teacher")


def _student_architecture(model: nn.Module) -> str:
    return validate_gipo_architecture(str(getattr(model, "architecture", "")), role="student")


def _teacher_target_spec_by_utility() -> Dict[str, MetricObjectiveSpec]:
    return {
        spec.utility_key: spec
        for specs in (FORECAST_METRIC_SPECS, CONDITIONAL_METRIC_SPECS, MOLECULE_METRIC_SPECS)
        for spec in specs
    }


def _truthy_row_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _finite_target_value(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def _target_component_applicable(row: MetricRow, target_key: str) -> bool:
    spec = _teacher_target_spec_by_utility().get(str(target_key))
    if spec is None or not spec.applicable_key:
        return True
    value = row.get(spec.applicable_key)
    if value in (None, ""):
        return False
    return _truthy_row_value(value)


def _row_reward_metric_weights(row: MetricRow, target_keys: Sequence[str]) -> Dict[str, float]:
    raw = row.get("reward_metric_weights_json", "")
    if raw in (None, ""):
        return {}
    try:
        payload = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid reward_metric_weights_json for context {context_id_from_row(row)!r}.") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("reward_metric_weights_json must decode to an object.")
    out: Dict[str, float] = {}
    for key in validate_teacher_metric_target_keys(target_keys):
        alias = _metric_weight_alias(key)
        if key in payload:
            value = float(payload[key])
        elif alias in payload:
            value = float(payload[alias])
        else:
            continue
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("reward_metric_weights_json values must be finite and nonnegative.")
        out[key] = value
    return out


def _masked_normalize_teacher_weights(raw_weights: torch.Tensor, target_mask: torch.Tensor | None) -> torch.Tensor:
    weights = raw_weights
    if target_mask is not None:
        if target_mask.shape != raw_weights.shape:
            raise ValueError("Teacher target mask must match metric weight shape.")
        weights = raw_weights * target_mask.to(device=raw_weights.device, dtype=raw_weights.dtype)
    denom = torch.sum(weights, dim=-1, keepdim=True)
    if bool(torch.any(denom <= 0.0).detach().cpu().item()):
        raise ValueError("Each teacher target row must have at least one positive-weight valid metric component.")
    return weights / denom


def _teacher_metric_weights(
    rows: Sequence[MetricRow] | None,
    *,
    target_keys: Sequence[str],
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    teacher_utility_weights: Mapping[str, float] | None = None,
    target_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    keys = validate_teacher_metric_target_keys(target_keys)
    if rows is not None:
        if len(rows) != int(batch):
            raise ValueError("Teacher scalarization rows must match the score batch length.")
        values: List[List[float]] = []
        default_weights = objective_weight_map_for_keys(keys)
        explicit_weights = normalize_teacher_utility_weights(keys, teacher_utility_weights) if teacher_utility_weights is not None else {}
        for row in rows:
            reward_weights = _row_reward_metric_weights(row, keys)
            if reward_weights:
                row_weights = {key: float(reward_weights.get(key, 0.0)) for key in keys}
            elif explicit_weights:
                row_weights = {key: float(explicit_weights[key]) for key in keys}
            else:
                row_weights = {}
                for key in keys:
                    alias = _metric_weight_alias(key)
                    row_weights[key] = float(
                        row.get(
                            f"{key}_weight",
                            row.get(
                                f"{alias}_weight",
                                row.get(f"u_comp_{alias}_weight", default_weights.get(key, 1.0)),
                            ),
                        )
                    )
            values.append([float(row_weights[key]) for key in keys])
        raw_weights = np.asarray(values, dtype=np.float32)
    elif teacher_utility_weights is not None:
        normalized = normalize_teacher_utility_weights(keys, teacher_utility_weights)
        raw_weights = np.tile(np.asarray([normalized[key] for key in keys], dtype=np.float32), (int(batch), 1))
    else:
        normalized = normalize_teacher_utility_weights(keys)
        raw_weights = np.tile(np.asarray([normalized[key] for key in keys], dtype=np.float32), (int(batch), 1))
    weights_tensor = torch.tensor(raw_weights, dtype=dtype, device=device)
    return _masked_normalize_teacher_weights(weights_tensor, target_mask)


def _teacher_metric_targets(
    rows: Sequence[MetricRow],
    *,
    target_keys: Sequence[str],
    device: torch.device | str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    keys = validate_teacher_metric_target_keys(target_keys)
    values: List[Tuple[float, ...]] = []
    masks: List[Tuple[float, ...]] = []
    for row in rows:
        row_values: List[float] = []
        row_mask: List[float] = []
        for key in keys:
            value = _finite_target_value(row.get(key))
            valid = value is not None and _target_component_applicable(row, key)
            row_values.append(float(value) if valid else 0.0)
            row_mask.append(1.0 if valid else 0.0)
        if not any(mask > 0.0 for mask in row_mask):
            raise ValueError(
                "Teacher metric target row has no valid utility components "
                f"for context={context_id_from_row(row)!r}, scheduler={row.get('scheduler_key')!r}."
            )
        values.append(tuple(row_values))
        masks.append(tuple(row_mask))
    return (
        torch.tensor(np.asarray(values, dtype=np.float32), dtype=torch.float32, device=device),
        torch.tensor(np.asarray(masks, dtype=np.float32), dtype=torch.float32, device=device),
    )


def _scalarize_teacher_metric_values(
    values: torch.Tensor,
    weights: torch.Tensor,
    *,
    target_keys: Sequence[str],
    target_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    expected = len(validate_teacher_metric_target_keys(target_keys))
    if values.ndim != 2 or values.shape[-1] != expected:
        raise ValueError(
            f"Teacher metric scores must have shape [batch, {expected}], got {tuple(values.shape)}."
        )
    if weights.shape != values.shape:
        raise ValueError("Teacher metric scalarization weights must match metric score shape.")
    normalized = _masked_normalize_teacher_weights(weights.to(device=values.device, dtype=values.dtype), target_mask)
    return torch.sum(values * normalized, dim=-1)


def _masked_smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape or target_mask.shape != target.shape:
        raise ValueError("Masked teacher regression tensors must have matching shapes.")
    loss = F.smooth_l1_loss(pred, target, reduction="none") * target_mask.to(device=pred.device, dtype=pred.dtype)
    denom = torch.sum(target_mask.to(device=pred.device, dtype=pred.dtype))
    if bool((denom <= 0.0).detach().cpu().item()):
        raise ValueError("Teacher regression mask has no valid components.")
    return torch.sum(loss) / denom


def _masked_metric_huber_values(pred: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> List[float | None]:
    if pred.shape != target.shape or target_mask.shape != target.shape:
        raise ValueError("Masked teacher regression tensors must have matching shapes.")
    loss = F.smooth_l1_loss(pred, target, reduction="none") * target_mask.to(device=pred.device, dtype=pred.dtype)
    counts = torch.sum(target_mask.to(device=pred.device, dtype=pred.dtype), dim=0)
    totals = torch.sum(loss, dim=0)
    out: List[float | None] = []
    for total, count in zip(totals.detach().cpu().tolist(), counts.detach().cpu().tolist()):
        out.append(None if float(count) <= 0.0 else float(total) / float(count))
    return out


def teacher_metric_target_keys_for_model(teacher: nn.Module) -> Tuple[str, ...]:
    _teacher_architecture(teacher)
    keys = getattr(teacher, "teacher_metric_targets", None)
    if keys is None and hasattr(teacher, "model_config"):
        keys = dict(teacher.model_config()).get("teacher_metric_targets")
    return validate_teacher_metric_target_keys(keys)


def _teacher_metric_scores(
    teacher: nn.Module,
    setting_features_batch: torch.Tensor,
    density_features_batch: torch.Tensor,
    series_index_batch: torch.Tensor,
    context_embedding_batch: torch.Tensor,
    *,
    rows: Sequence[MetricRow] | None = None,
) -> torch.Tensor:
    _teacher_architecture(teacher)
    target_keys = teacher_metric_target_keys_for_model(teacher)
    scores = teacher(
        setting_features_batch,
        density_features_batch,
        series_index_batch,
        context_embedding_batch,
        rows=rows,
    )
    if scores.ndim != 2 or scores.shape[-1] != len(target_keys):
        raise ValueError(
            f"GIPO teacher must return a metric vector with {len(target_keys)} columns; got {tuple(scores.shape)}."
        )
    return scores


def _teacher_scores(
    teacher: nn.Module,
    setting_features_batch: torch.Tensor,
    density_features_batch: torch.Tensor,
    series_index_batch: torch.Tensor,
    context_embedding_batch: torch.Tensor,
    *,
    rows: Sequence[MetricRow] | None = None,
    teacher_utility_weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    target_keys = teacher_metric_target_keys_for_model(teacher)
    metric_scores = _teacher_metric_scores(
        teacher,
        setting_features_batch,
        density_features_batch,
        series_index_batch,
        context_embedding_batch,
        rows=rows,
    )
    target_mask = None
    if rows is not None:
        _, target_mask = _teacher_metric_targets(rows, target_keys=target_keys, device=metric_scores.device)
    weights = _teacher_metric_weights(
        rows,
        target_keys=target_keys,
        batch=int(setting_features_batch.shape[0]),
        device=metric_scores.device,
        dtype=metric_scores.dtype,
        teacher_utility_weights=teacher_utility_weights,
        target_mask=target_mask,
    )
    return _scalarize_teacher_metric_values(metric_scores, weights, target_keys=target_keys, target_mask=target_mask)


def _student_logits(
    student: nn.Module,
    setting_features_batch: torch.Tensor,
    series_index_batch: torch.Tensor,
    context_embedding_batch: torch.Tensor,
    *,
    rows: Sequence[MetricRow] | None = None,
) -> torch.Tensor:
    _student_architecture(student)
    return student.logits(
        setting_features_batch,
        series_index_batch,
        context_embedding_batch,
        rows=rows,
    )


def _student_density_mass(
    student: nn.Module,
    setting_features_batch: torch.Tensor,
    series_index_batch: torch.Tensor,
    context_embedding_batch: torch.Tensor,
    *,
    rows: Sequence[MetricRow] | None = None,
) -> torch.Tensor:
    return torch.softmax(
        _student_logits(
            student,
            setting_features_batch,
            series_index_batch,
            context_embedding_batch,
            rows=rows,
        ),
        dim=-1,
    )


def _pair_indices(
    targets: torch.Tensor,
    pair_keys: Sequence[ContextPairKey],
    *,
    margin: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = [float(value) for value in targets.detach().cpu().tolist()]
    by_key: Dict[ContextPairKey, List[int]] = defaultdict(list)
    for idx, key in enumerate(pair_keys):
        by_key[key].append(int(idx))
    left: List[int] = []
    right: List[int] = []
    signs: List[float] = []
    min_delta = float(margin)
    for indices in by_key.values():
        for pos, i in enumerate(indices):
            for j in indices[pos + 1 :]:
                diff = values[i] - values[j]
                if abs(diff) <= min_delta:
                    continue
                left.append(i)
                right.append(j)
                signs.append(1.0 if diff > 0.0 else -1.0)
    if not left:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )
    return (
        torch.tensor(left, dtype=torch.long, device=device),
        torch.tensor(right, dtype=torch.long, device=device),
        torch.tensor(signs, dtype=torch.float32, device=device),
    )


def pairwise_rank_loss(
    pred: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    sign: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if left.numel() == 0:
        return pred.new_zeros(())
    temp = _finite_positive(temperature, label="teacher_rank_temperature")
    logits = sign * (pred[left] - pred[right]) / temp
    return F.softplus(-logits).mean()


def _rank_values_desc(values: Mapping[str, float]) -> Dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (-float(item[1]), str(item[0])))
    return {str(key): float(idx) for idx, (key, _) in enumerate(ordered)}


def _spearman_for_scores(pred_by_schedule: Mapping[str, float], target_by_schedule: Mapping[str, float]) -> float:
    keys = sorted(set(pred_by_schedule) & set(target_by_schedule))
    if len(keys) < 2:
        return 0.0
    pred_ranks = _rank_values_desc({key: float(pred_by_schedule[key]) for key in keys})
    target_ranks = _rank_values_desc({key: float(target_by_schedule[key]) for key in keys})
    pred = np.asarray([pred_ranks[key] for key in keys], dtype=np.float64)
    target = np.asarray([target_ranks[key] for key in keys], dtype=np.float64)
    pred_std = float(np.std(pred))
    target_std = float(np.std(target))
    if pred_std <= 1e-12 or target_std <= 1e-12:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1])


def _teacher_training_tensors(
    rows: Sequence[MetricRow],
    *,
    context_embeddings: Mapping[str, Sequence[float]],
    series_index_map: Mapping[str, int],
    schedule_grids: Mapping[ScheduleGridKey, Sequence[float]] | None,
    reference_time_grid: Sequence[float],
    density_normalizer: DensityFeatureNormalizer,
    device: torch.device | str,
    setting_feature_mode: str = SETTING_ENCODER_MODE_CONTINUOUS_V3,
    setting_encoder_config: Mapping[str, Any] | SettingEncoderConfig | None = None,
    teacher_metric_target_keys: Sequence[str] = TEACHER_METRIC_TARGET_KEYS,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[ContextPairKey], List[str], List[Tuple[float, ...]]]:
    feature_mode = validate_setting_feature_mode(setting_feature_mode)
    encoder_config = _resolve_setting_encoder_config(feature_mode, setting_encoder_config)
    setting_rows: List[torch.Tensor] = []
    density_rows: List[np.ndarray] = []
    series_rows: List[int] = []
    context_rows: List[np.ndarray] = []
    pair_keys: List[ContextPairKey] = []
    schedule_keys: List[str] = []
    density_masses: List[Tuple[float, ...]] = []
    for row in rows:
        context_embedding_id = context_embedding_id_from_row(row)
        if context_embedding_id not in context_embeddings:
            raise KeyError(f"Missing context embedding for {context_embedding_id}.")
        solver = normalize_solver_key(str(row["solver_key"]))
        target_nfe = int(row["target_nfe"])
        mass = density_mass_for_row(row, schedule_grids=schedule_grids, reference_time_grid=reference_time_grid)
        setting_rows.append(setting_features(solver, target_nfe, mode=feature_mode, config=encoder_config))
        density_rows.append(density_normalizer.transform_one(mass, reference_time_grid=reference_time_grid))
        series_rows.append(0)
        context_rows.append(np.asarray(context_embeddings[context_embedding_id], dtype=np.float32))
        pair_keys.append(context_pair_key(row, pair_on_seed=True))
        schedule_keys.append(str(row["scheduler_key"]))
        density_masses.append(mass)
    metric_targets, metric_target_mask = _teacher_metric_targets(rows, target_keys=teacher_metric_target_keys, device=device)
    return (
        torch.stack(setting_rows, dim=0).to(device=device),
        torch.tensor(np.stack(density_rows, axis=0), dtype=torch.float32, device=device),
        torch.tensor(series_rows, dtype=torch.long, device=device),
        torch.tensor(np.stack(context_rows, axis=0), dtype=torch.float32, device=device),
        metric_targets,
        metric_target_mask,
        pair_keys,
        schedule_keys,
        density_masses,
    )


def gipo_teacher_diagnostics(
    teacher: nn.Module,
    rows: Sequence[MetricRow],
    *,
    context_embeddings: Mapping[str, Sequence[float]],
    series_index_map: Mapping[str, int],
    schedule_grids: Mapping[ScheduleGridKey, Sequence[float]] | None,
    reference_time_grid: Sequence[float],
    density_normalizer: DensityFeatureNormalizer,
    rank_temperature: float = 0.5,
    regression_weight: float = 0.25,
    pair_margin: float = 0.0,
    pair_on_seed: bool = True,
    split_name: str = "diagnostic",
    fit_context_ids: Iterable[str] = (),
    fit_series_keys: Iterable[str] = (),
    setting_feature_mode: str = SETTING_ENCODER_MODE_CONTINUOUS_V3,
    setting_encoder_config: Mapping[str, Any] | SettingEncoderConfig | None = None,
    teacher_utility_weights: Mapping[str, float] | None = None,
    complete_candidate_schedule_keys: Sequence[str] | None = None,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    feature_mode = validate_setting_feature_mode(setting_feature_mode)
    encoder_config = _resolve_setting_encoder_config(feature_mode, setting_encoder_config)
    rank_temperature, regression_weight, pair_margin = validate_teacher_objective_hyperparameters(
        rank_temperature=rank_temperature,
        regression_weight=regression_weight,
        pair_margin=pair_margin,
    )
    selection_temperature = DEFAULT_TEACHER_TARGET_TEMPERATURE
    contexts = {context_id_from_row(row) for row in rows}
    series_keys = {series_key_from_row(row) for row in rows}
    schedules = {str(row["scheduler_key"]) for row in rows}
    expected_candidate_schedules = (
        tuple(str(key) for key in complete_candidate_schedule_keys)
        if complete_candidate_schedule_keys is not None
        else tuple(sorted(schedules))
    )
    fit_context_set = {str(value) for value in fit_context_ids}
    fit_series_set = {str(value) for value in fit_series_keys}
    base: Dict[str, Any] = {
        "split_name": str(split_name),
        "row_count": int(len(rows)),
        "context_count": int(len(contexts)),
        "series_count": int(len(series_keys)),
        "schedule_count": int(len(schedules)),
        "schedule_keys": sorted(schedules),
        "series_conditioning": SERIES_CONDITIONING_NONE_CONTEXT_ONLY,
        "split_phases": sorted({str(row.get("split_phase", row.get("split", ""))) for row in rows}),
        "fit_context_overlap_count": int(len(contexts & fit_context_set)),
        "fit_series_overlap_count": int(len(series_keys & fit_series_set)),
        "setting_feature_mode": feature_mode,
        "uses_validation_labels": False,
        "complete_candidate_group_key_schema": [
            "context_id",
            "solver_key",
            "target_nfe",
            "realized_nfe",
            "logical_seed",
            "evaluation_seed",
        ],
        "complete_candidate_schedule_keys": list(expected_candidate_schedules),
    }
    if not rows:
        base.update(
            {
                "rank_loss": None,
                "huber_loss": None,
                "total_loss": None,
                "pairwise_accuracy": None,
                "pair_count": 0,
                "spearman_rank_correlation": None,
                "best_candidate_agreement": None,
                "candidate_group_count": 0,
                "complete_candidate_group_count": 0,
                "soft_regret": None,
                "top1_regret": None,
                "validation_soft_regret": None,
                "validation_top1_regret": None,
            }
        )
        return base
    teacher_metric_target_keys = teacher_metric_target_keys_for_model(teacher)
    was_training = bool(teacher.training)
    teacher.eval()
    with torch.no_grad():
        sx, dx, series_idx, cx, metric_targets, metric_target_mask, pair_keys, schedule_keys, _ = _teacher_training_tensors(
            rows,
            context_embeddings=context_embeddings,
            series_index_map=series_index_map,
            schedule_grids=schedule_grids,
            reference_time_grid=reference_time_grid,
            density_normalizer=density_normalizer,
            setting_feature_mode=feature_mode,
            setting_encoder_config=encoder_config,
            teacher_metric_target_keys=teacher_metric_target_keys,
            device=device,
        )
        target_weights = _teacher_metric_weights(
            rows,
            target_keys=teacher_metric_target_keys,
            batch=len(rows),
            device=sx.device,
            dtype=metric_targets.dtype,
            teacher_utility_weights=teacher_utility_weights,
            target_mask=metric_target_mask,
        )
        targets = _scalarize_teacher_metric_values(
            metric_targets,
            target_weights,
            target_keys=teacher_metric_target_keys,
            target_mask=metric_target_mask,
        )
        if not pair_on_seed:
            pair_keys = [key[:4] + (None,) for key in pair_keys]
        rank_left, rank_right, rank_sign = _pair_indices(targets, pair_keys, margin=float(pair_margin), device=sx.device)
        metric_pred = _teacher_metric_scores(teacher, sx, dx, series_idx, cx, rows=rows)
        pred = _scalarize_teacher_metric_values(
            metric_pred,
            target_weights,
            target_keys=teacher_metric_target_keys,
            target_mask=metric_target_mask,
        )
        rank = pairwise_rank_loss(pred, rank_left, rank_right, rank_sign, temperature=float(rank_temperature))
        huber = _masked_smooth_l1_loss(metric_pred, metric_targets, metric_target_mask)
        total = rank + float(regression_weight) * huber
        if rank_left.numel() > 0:
            predicted_sign = torch.sign(pred[rank_left] - pred[rank_right])
            pairwise_accuracy = float((predicted_sign == rank_sign).to(torch.float32).mean().detach().cpu().item())
        else:
            pairwise_accuracy = None
        pred_values = [float(value) for value in pred.detach().cpu().tolist()]
        target_values = [float(value) for value in targets.detach().cpu().tolist()]
        metric_huber_values = _masked_metric_huber_values(metric_pred, metric_targets, metric_target_mask)
    if was_training:
        teacher.train()

    expected_schedule_set = set(expected_candidate_schedules)
    candidate_groups: Dict[Tuple[str, str, int, int | None, int | None, int | None], Dict[str, List[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    unexpected_schedule_groups = 0
    for idx, row in enumerate(rows):
        schedule = str(schedule_keys[idx])
        if expected_schedule_set and schedule not in expected_schedule_set:
            unexpected_schedule_groups += 1
            continue
        candidate_groups[teacher_selection_candidate_group_key(row)][schedule].append(int(idx))
    spearman_values: List[float] = []
    best_candidate_hits = 0
    best_candidate_total = 0
    soft_regret_values: List[float] = []
    top1_regret_values: List[float] = []
    oracle_utility_values: List[float] = []
    teacher_expected_utility_values: List[float] = []
    selected_utility_values: List[float] = []
    incomplete_candidate_groups = 0
    duplicate_schedule_groups = 0
    missing_schedule_groups = 0
    complete_group_examples: List[Dict[str, Any]] = []
    for group_key, schedule_to_indices in candidate_groups.items():
        missing_schedules = [
            schedule
            for schedule in expected_candidate_schedules
            if len(schedule_to_indices.get(schedule, [])) != 1
        ]
        if missing_schedules:
            incomplete_candidate_groups += 1
            missing_schedule_groups += 1
            continue
        if any(len(indices) != 1 for indices in schedule_to_indices.values()):
            duplicate_schedule_groups += 1
            continue
        ordered = list(expected_candidate_schedules)
        if len(ordered) < 2:
            incomplete_candidate_groups += 1
            continue
        group_indices = [schedule_to_indices[schedule][0] for schedule in ordered]
        group_pred = np.asarray([pred_values[idx] for idx in group_indices], dtype=np.float64)
        group_target = np.asarray([target_values[idx] for idx in group_indices], dtype=np.float64)
        shifted = group_pred / float(selection_temperature)
        shifted = shifted - float(np.max(shifted))
        exp_values = np.exp(shifted)
        weights = exp_values / max(float(np.sum(exp_values)), 1e-12)
        oracle_utility = float(np.max(group_target))
        top_idx = sorted(range(len(ordered)), key=lambda pos: (-float(group_pred[pos]), ordered[pos]))[0]
        oracle_idx = sorted(range(len(ordered)), key=lambda pos: (-float(group_target[pos]), ordered[pos]))[0]
        selected_utility = float(group_target[top_idx])
        expected_utility = float(np.dot(weights, group_target))
        top1_regret = float(max(0.0, oracle_utility - selected_utility))
        soft_regret = float(max(0.0, oracle_utility - expected_utility))
        top1_regret_values.append(top1_regret)
        soft_regret_values.append(soft_regret)
        oracle_utility_values.append(oracle_utility)
        teacher_expected_utility_values.append(expected_utility)
        selected_utility_values.append(selected_utility)
        spearman_values.append(
            _spearman_for_scores(
                {schedule: float(group_pred[pos]) for pos, schedule in enumerate(ordered)},
                {schedule: float(group_target[pos]) for pos, schedule in enumerate(ordered)},
            )
        )
        best_candidate_hits += int(top_idx == oracle_idx)
        best_candidate_total += 1
        if len(complete_group_examples) < 16:
            complete_group_examples.append(
                {
                    **complete_candidate_group_payload(group_key),
                    "teacher_top_schedule_key": str(ordered[top_idx]),
                    "oracle_top_schedule_key": str(ordered[oracle_idx]),
                    "soft_regret": soft_regret,
                    "top1_regret": top1_regret,
                }
            )
    sequence_delta_values: List[float] = []
    sequence_second_diff_values: List[float] = []
    by_candidate_sequence: Dict[Tuple[str, int | None, str, str, str, str], List[Tuple[int, int]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        sequence_key = (*_nfe_sequence_key(row), str(schedule_keys[idx]))
        by_candidate_sequence[sequence_key].append((int(row["target_nfe"]), int(idx)))
    for items in by_candidate_sequence.values():
        ordered = sorted(items, key=lambda item: (int(item[0]), int(item[1])))
        values = [float(pred_values[idx]) for _, idx in ordered]
        for left_value, right_value in zip(values, values[1:]):
            sequence_delta_values.append(abs(float(right_value) - float(left_value)))
        if len(values) >= 3:
            for a, b, c in zip(values, values[1:], values[2:]):
                sequence_second_diff_values.append(abs(float(c) - 2.0 * float(b) + float(a)))
    delta_stats = _summary_percentiles(sequence_delta_values)
    second_stats = _summary_percentiles(sequence_second_diff_values)
    soft_regret_stats = _summary_percentiles(soft_regret_values)
    top1_regret_stats = _summary_percentiles(top1_regret_values)
    base.update(
        {
            "rank_loss": float(rank.detach().cpu().item()),
            "huber_loss": float(huber.detach().cpu().item()),
            **{
                f"metric_huber_loss_{_metric_weight_alias(key)}": (None if value is None else float(value))
                for key, value in zip(teacher_metric_target_keys, metric_huber_values)
            },
            "total_loss": float(total.detach().cpu().item()),
            "pairwise_accuracy": pairwise_accuracy,
            "pair_count": int(rank_left.numel()),
            "spearman_rank_correlation": float(np.mean(np.asarray(spearman_values, dtype=np.float64))) if spearman_values else None,
            "best_candidate_agreement": None if best_candidate_total == 0 else float(best_candidate_hits / best_candidate_total),
            "candidate_group_count": int(best_candidate_total),
            "complete_candidate_group_count": int(len(soft_regret_values)),
            "complete_candidate_group_raw_count": int(len(candidate_groups)),
            "complete_candidate_group_incomplete_count": int(incomplete_candidate_groups),
            "complete_candidate_group_missing_schedule_count": int(missing_schedule_groups),
            "complete_candidate_group_duplicate_schedule_count": int(duplicate_schedule_groups),
            "complete_candidate_group_unexpected_schedule_row_count": int(unexpected_schedule_groups),
            "complete_candidate_group_regret_examples": complete_group_examples,
            "selection_candidate_group_count": int(len(soft_regret_values)),
            "selection_candidate_group_raw_count": int(len(candidate_groups)),
            "selection_candidate_incomplete_group_count": int(incomplete_candidate_groups),
            "selection_candidate_duplicate_schedule_group_count": int(duplicate_schedule_groups),
            "soft_regret": None if not soft_regret_values else soft_regret_stats["mean"],
            "soft_regret_p95": None if not soft_regret_values else soft_regret_stats["p95"],
            "top1_regret": None if not top1_regret_values else top1_regret_stats["mean"],
            "top1_regret_p95": None if not top1_regret_values else top1_regret_stats["p95"],
            "validation_soft_regret": None if not soft_regret_values else soft_regret_stats["mean"],
            "validation_soft_regret_p95": None if not soft_regret_values else soft_regret_stats["p95"],
            "validation_top1_regret": None if not top1_regret_values else top1_regret_stats["mean"],
            "validation_top1_regret_p95": None if not top1_regret_values else top1_regret_stats["p95"],
            "validation_oracle_utility_mean": _summary_percentiles(oracle_utility_values)["mean"],
            "validation_teacher_expected_utility_mean": _summary_percentiles(teacher_expected_utility_values)["mean"],
            "validation_teacher_top1_utility_mean": _summary_percentiles(selected_utility_values)["mean"],
            "teacher_nfe_sequence_candidate_count": int(len(by_candidate_sequence)),
            "teacher_nfe_adjacent_abs_utility_delta_mean": delta_stats["mean"],
            "teacher_nfe_adjacent_abs_utility_delta_p95": delta_stats["p95"],
            "teacher_nfe_second_difference_abs_mean": second_stats["mean"],
            "teacher_nfe_second_difference_abs_p95": second_stats["p95"],
        }
    )
    return base


def _selected_gipo_teacher_checkpoint(
    checkpoint_history: Sequence[Dict[str, Any]],
    checkpoint_states: Mapping[int, Mapping[str, torch.Tensor]],
    *,
    required_split_names: Sequence[str] = (),
    component_weights: Mapping[str, float] | None = None,
) -> Tuple[Dict[str, Any], Mapping[str, torch.Tensor] | None]:
    mode = TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET

    def _component_weight(name: str, weights: Mapping[str, float]) -> float:
        split = str(name)
        if split in weights:
            return float(weights[split])
        if split.startswith("context"):
            return float(weights.get("context", weights.get(split, 0.0)))
        if split.startswith("density_family"):
            return float(weights.get("density_family", weights.get(split, 0.0)))
        if split.startswith("unseen_nfe"):
            return float(weights.get("unseen_nfe", weights.get(split, 0.0)))
        return float(weights.get(split, 0.0))

    if not checkpoint_history:
        raise ValueError("Weighted normalized regret selection requires checkpoint diagnostics.")

    required = tuple(str(name) for name in required_split_names)
    scored: List[Dict[str, Any]] = []
    required_metric_names = ("validation_soft_regret",)
    for entry in checkpoint_history:
        diagnostics = {str(key): dict(value) for key, value in dict(entry.get("diagnostics", {})).items()}
        missing: List[str] = []
        for name in required:
            diag = diagnostics.get(name, {})
            for metric_name in required_metric_names:
                if diag.get(metric_name) is None:
                    missing.append(f"{name}:missing_{metric_name}")
            if int(diag.get("complete_candidate_group_count", 0) or 0) <= 0:
                missing.append(f"{name}:no_complete_candidate_groups")
        if missing:
            copied = dict(entry)
            copied["selection_constraints_passed"] = False
            copied["selection_constraint_failures"] = missing
            scored.append(copied)
            continue
        active_names = required or tuple(
            sorted(
                name
                for name, diag in diagnostics.items()
                if all(diag.get(metric_name) is not None for metric_name in required_metric_names)
            )
        )
        if not active_names:
            copied = dict(entry)
            copied["selection_constraints_passed"] = False
            copied["selection_constraint_failures"] = ["no_active_diagnostic_splits"]
            scored.append(copied)
            continue
        losses = [
            float(diagnostics[name]["total_loss"])
            for name in active_names
            if diagnostics[name].get("total_loss") is not None
        ]
        rank_values = [float(diagnostics[name]["rank_loss"]) for name in active_names if diagnostics[name].get("rank_loss") is not None]
        huber_values = [float(diagnostics[name]["huber_loss"]) for name in active_names if diagnostics[name].get("huber_loss") is not None]
        pairwise_values = [float(diagnostics[name]["pairwise_accuracy"]) for name in active_names if diagnostics[name].get("pairwise_accuracy") is not None]
        spearman_values = [float(diagnostics[name]["spearman_rank_correlation"]) for name in active_names if diagnostics[name].get("spearman_rank_correlation") is not None]
        agreement_values = [float(diagnostics[name]["best_candidate_agreement"]) for name in active_names if diagnostics[name].get("best_candidate_agreement") is not None]
        soft_regret_values = [float(diagnostics[name]["validation_soft_regret"]) for name in active_names if diagnostics[name].get("validation_soft_regret") is not None]
        top1_regret_values = [float(diagnostics[name]["validation_top1_regret"]) for name in active_names if diagnostics[name].get("validation_top1_regret") is not None]
        copied = dict(entry)
        copied["diagnostics"] = diagnostics
        copied["selection_constraints_passed"] = True
        copied["selection_constraint_failures"] = []
        copied["mean_diagnostic_total_loss"] = float(np.mean(np.asarray(losses, dtype=np.float64))) if losses else 0.0
        copied["mean_rank_loss"] = float(np.mean(np.asarray(rank_values, dtype=np.float64))) if rank_values else 0.0
        copied["mean_huber_loss"] = float(np.mean(np.asarray(huber_values, dtype=np.float64))) if huber_values else 0.0
        copied["mean_pairwise_accuracy"] = float(np.mean(np.asarray(pairwise_values, dtype=np.float64))) if pairwise_values else 0.0
        copied["mean_spearman_rank_correlation"] = float(np.mean(np.asarray(spearman_values, dtype=np.float64))) if spearman_values else 0.0
        copied["mean_best_candidate_agreement"] = float(np.mean(np.asarray(agreement_values, dtype=np.float64))) if agreement_values else 0.0
        copied["mean_validation_soft_regret"] = float(np.mean(np.asarray(soft_regret_values, dtype=np.float64))) if soft_regret_values else None
        copied["mean_validation_top1_regret"] = float(np.mean(np.asarray(top1_regret_values, dtype=np.float64))) if top1_regret_values else None
        copied["selection_active_splits"] = list(active_names)
        scored.append(copied)

    valid_entries = [entry for entry in scored if bool(entry.get("selection_constraints_passed", False))]
    if not valid_entries:
        raise ValueError("Weighted normalized regret selection found no valid checkpoint candidates.")
    active_splits = tuple(required or tuple(valid_entries[0].get("selection_active_splits", ()) or ()))
    if not active_splits:
        raise ValueError("Weighted normalized regret selection found no active diagnostic splits.")

    raw_by_split: Dict[str, Dict[int, float]] = {}
    for name in active_splits:
        split_values: Dict[int, float] = {}
        for entry in valid_entries:
            diag = dict(dict(entry.get("diagnostics", {}) or {}).get(name, {}) or {})
            if diag.get("validation_soft_regret") is None:
                raise ValueError(f"Weighted normalized regret selection missing validation_soft_regret for split {name!r}.")
            split_values[int(entry["step"])] = float(diag["validation_soft_regret"])
        raw_by_split[name] = split_values

    ranges: Dict[str, Dict[str, float]] = {}
    normalized_by_split: Dict[str, Dict[int, float]] = {}
    for name, values in raw_by_split.items():
        lo = min(float(value) for value in values.values())
        hi = max(float(value) for value in values.values())
        ranges[name] = {"min": float(lo), "max": float(hi), "span": float(hi - lo)}
        if hi - lo <= 1e-12:
            normalized_by_split[name] = {int(step): 0.0 for step in values}
        else:
            normalized_by_split[name] = {
                int(step): (float(value) - lo) / (hi - lo)
                for step, value in values.items()
            }

    if component_weights:
        raw_weights = {name: max(0.0, _component_weight(name, dict(component_weights or {}))) for name in active_splits}
    else:
        raw_weights = {
            name: (
                0.25
                if str(name).startswith("context") or str(name).startswith("density_family")
                else 0.50
                if str(name).startswith("unseen_nfe")
                else 0.0
            )
            for name in active_splits
        }
    weight_sum = sum(float(value) for value in raw_weights.values())
    if weight_sum <= 0.0:
        raw_weights = {name: 1.0 for name in active_splits}
        weight_sum = float(len(active_splits))
    normalized_weights = {name: float(value) / weight_sum for name, value in raw_weights.items()}

    for entry in valid_entries:
        step = int(entry["step"])
        raw_values = {name: float(raw_by_split[name][step]) for name in active_splits}
        normalized_values = {name: float(normalized_by_split[name][step]) for name in active_splits}
        weighted_score = float(sum(float(normalized_weights[name]) * float(normalized_values[name]) for name in active_splits))
        minimax_score = float(max(normalized_values.values())) if normalized_values else 0.0
        raw_mean = float(np.mean(np.asarray(list(raw_values.values()), dtype=np.float64))) if raw_values else 0.0
        entry["selection_raw_regret_values"] = dict(raw_values)
        entry["selection_normalized_regret_values"] = dict(normalized_values)
        entry["selection_component_values"] = dict(normalized_values)
        entry["selection_component_ranges"] = dict(ranges)
        entry["selection_component_weights"] = dict(normalized_weights)
        entry["weighted_normalized_regret_score"] = weighted_score
        entry["checkpoint_selection_score"] = weighted_score
        entry["minimax_normalized_regret"] = minimax_score
        entry["raw_mean_validation_soft_regret"] = raw_mean

    best_entry = min(
        valid_entries,
        key=lambda entry: (
            float(entry["weighted_normalized_regret_score"]),
            float(entry["minimax_normalized_regret"]),
            float(entry["raw_mean_validation_soft_regret"]),
            int(entry["step"]),
        ),
    )
    selected_step = int(best_entry["step"])
    return (
        {
            "selection_protocol": mode,
            "selection_split": "teacher_holdouts",
            "selection_metric": "weighted_normalized_regret_score",
            "selection_mode": mode,
            "selected_step": selected_step,
            "selected_mean_diagnostic_total_loss": best_entry.get("mean_diagnostic_total_loss"),
            "selected_mean_validation_soft_regret": best_entry.get("mean_validation_soft_regret"),
            "selected_mean_validation_top1_regret": best_entry.get("mean_validation_top1_regret"),
            "selected_mean_best_candidate_agreement": best_entry.get("mean_best_candidate_agreement"),
            "selected_mean_spearman_rank_correlation": best_entry.get("mean_spearman_rank_correlation"),
            "selected_mean_rank_loss": best_entry.get("mean_rank_loss"),
            "selected_mean_huber_loss": best_entry.get("mean_huber_loss"),
            "selected_weighted_normalized_regret_score": best_entry.get("weighted_normalized_regret_score"),
            "selected_normalized_regret_values": best_entry.get("selection_normalized_regret_values", {}),
            "selected_raw_regret_values": best_entry.get("selection_raw_regret_values", {}),
            "selected_regret_normalization_ranges": best_entry.get("selection_component_ranges", {}),
            "selected_minimax_normalized_regret": best_entry.get("minimax_normalized_regret"),
            "selected_raw_mean_validation_soft_regret": best_entry.get("raw_mean_validation_soft_regret"),
            "selected_checkpoint_selection_score": best_entry.get("checkpoint_selection_score"),
            "selected_component_values": best_entry.get("selection_component_values", {}),
            "selected_component_weights": best_entry.get("selection_component_weights", {}),
            "tie_breaker": "weighted_score_then_minimax_normalized_regret_then_raw_mean_regret_then_earlier_step",
            "uses_validation_labels": False,
            "locked_test_used_for_selection": False,
            "history": scored,
        },
        checkpoint_states.get(selected_step),
    )


def select_weighted_normalized_regret_checkpoint(
    checkpoint_history: Sequence[Dict[str, Any]],
    *,
    required_split_names: Sequence[str] = (),
    component_weights: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    selection, _ = _selected_gipo_teacher_checkpoint(
        checkpoint_history,
        {},
        required_split_names=required_split_names,
        component_weights=component_weights,
    )
    return selection


def _should_log_step(step_index: int, steps: int, log_every: int | None) -> bool:
    total_steps = int(steps)
    step_value = int(step_index) + 1
    if step_value == 1 or step_value == total_steps:
        return True
    cadence = int(log_every or 0)
    if cadence <= 0:
        cadence = max(1, total_steps // 5)
    return step_value % max(1, cadence) == 0


def _loss_tail_slope(losses: Sequence[Mapping[str, Any]], key: str, *, tail_count: int = 5) -> float | None:
    points = [
        (float(entry["step"]), float(entry[key]))
        for entry in list(losses)[-max(2, int(tail_count)) :]
        if entry.get("step") is not None and entry.get(key) is not None
    ]
    if len(points) < 2:
        return None
    x = np.asarray([point[0] for point in points], dtype=np.float64)
    y = np.asarray([point[1] for point in points], dtype=np.float64)
    x = x - float(np.mean(x))
    denom = float(np.sum(x * x))
    if denom <= 0.0:
        return None
    return float(np.sum(x * (y - float(np.mean(y)))) / denom)


def train_gipo_teacher(
    teacher: nn.Module,
    rows: Sequence[MetricRow],
    *,
    context_embeddings: Mapping[str, Sequence[float]],
    series_index_map: Mapping[str, int],
    schedule_grids: Mapping[ScheduleGridKey, Sequence[float]] | None,
    reference_time_grid: Sequence[float],
    density_normalizer: DensityFeatureNormalizer,
    steps: int = 500,
    lr: float = 1e-3,
    rank_temperature: float = 0.5,
    regression_weight: float = 0.25,
    pair_margin: float = 0.0,
    pair_on_seed: bool = True,
    require_rank_pairs: bool = True,
    diagnostic_splits: Mapping[str, Sequence[MetricRow]] | None = None,
    diagnostic_candidate_schedule_keys: Mapping[str, Sequence[str]] | None = None,
    teacher_checkpoint_every: int = 100,
    teacher_loss_log_every: int = 0,
    teacher_selection_axis_weights: Mapping[str, float] | None = None,
    final_retrain_mode: bool = False,
    seed: int = 0,
    allowed_schedule_keys: Sequence[str] = DEFAULT_SUPERVISION_SCHEDULE_KEYS,
    setting_feature_mode: str = SETTING_ENCODER_MODE_CONTINUOUS_V3,
    setting_encoder_config: Mapping[str, Any] | SettingEncoderConfig | None = None,
    teacher_utility_weights: Mapping[str, float] | None = None,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Teacher training requires at least one context reward row.")
    validate_gipo_support_schedule_keys(sorted({str(row["scheduler_key"]) for row in rows}), allowed_schedule_keys=allowed_schedule_keys)
    feature_mode = validate_setting_feature_mode(setting_feature_mode)
    encoder_config = _resolve_setting_encoder_config(feature_mode, setting_encoder_config)
    rank_temperature, regression_weight, pair_margin = validate_teacher_objective_hyperparameters(
        rank_temperature=rank_temperature,
        regression_weight=regression_weight,
        pair_margin=pair_margin,
    )
    teacher_metric_target_keys = teacher_metric_target_keys_for_model(teacher)
    selection_mode = TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET
    selection_temperature = DEFAULT_TEACHER_TARGET_TEMPERATURE
    teacher.to(device)
    sx, dx, series_idx, cx, metric_targets, metric_target_mask, pair_keys, _, _ = _teacher_training_tensors(
        rows,
        context_embeddings=context_embeddings,
        series_index_map=series_index_map,
        schedule_grids=schedule_grids,
        reference_time_grid=reference_time_grid,
        density_normalizer=density_normalizer,
        device=device,
        setting_feature_mode=feature_mode,
        setting_encoder_config=encoder_config,
        teacher_metric_target_keys=teacher_metric_target_keys,
        seed=int(seed),
    )
    target_weights = _teacher_metric_weights(
        rows,
        target_keys=teacher_metric_target_keys,
        batch=len(rows),
        device=sx.device,
        dtype=metric_targets.dtype,
        teacher_utility_weights=teacher_utility_weights,
        target_mask=metric_target_mask,
    )
    targets = _scalarize_teacher_metric_values(
        metric_targets,
        target_weights,
        target_keys=teacher_metric_target_keys,
        target_mask=metric_target_mask,
    )
    if not pair_on_seed:
        pair_keys = [key[:4] + (None,) for key in pair_keys]
    left, right, sign = _pair_indices(targets, pair_keys, margin=float(pair_margin), device=sx.device)
    if require_rank_pairs and left.numel() == 0:
        raise ValueError("Context teacher training found no same-context schedule pairs for ranking.")
    opt = torch.optim.AdamW(teacher.parameters(), lr=float(lr), weight_decay=1e-4)
    losses: List[Dict[str, Any]] = []
    checkpoint_history: List[Dict[str, Any]] = []
    checkpoint_states: Dict[int, Mapping[str, torch.Tensor]] = {}
    fit_context_ids = sorted({context_id_from_row(row) for row in rows})
    fit_series_keys = sorted({series_key_from_row(row) for row in rows})
    diagnostic_candidate_schedule_map = {
        str(name): tuple(str(key) for key in keys)
        for name, keys in dict(diagnostic_candidate_schedule_keys or {}).items()
    }
    checkpoint_every = max(1, int(teacher_checkpoint_every))
    for step in range(int(steps)):
        metric_pred = _teacher_metric_scores(teacher, sx, dx, series_idx, cx, rows=rows)
        pred = _scalarize_teacher_metric_values(
            metric_pred,
            target_weights,
            target_keys=teacher_metric_target_keys,
            target_mask=metric_target_mask,
        )
        rank = pairwise_rank_loss(pred, left, right, sign, temperature=float(rank_temperature))
        huber = _masked_smooth_l1_loss(metric_pred, metric_targets, metric_target_mask)
        loss = rank + float(regression_weight) * huber
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if _should_log_step(step, int(steps), int(teacher_loss_log_every)):
            losses.append(
                {
                    "step": int(step + 1),
                    "teacher_total_loss": float(loss.detach().cpu().item()),
                    "teacher_rank_loss": float(rank.detach().cpu().item()),
                    "teacher_huber_loss": float(huber.detach().cpu().item()),
                    "teacher_pair_count": int(left.numel()),
                    "teacher_target": "metric_vector",
                    "teacher_scalar_target": "weighted_metric_utility",
                    "reward_anchor_schedule_key": UNIFORM_SCHEDULE_KEY,
                }
            )
        if diagnostic_splits and (step == 0 or step == int(steps) - 1 or (step + 1) % checkpoint_every == 0):
            diagnostics = {
                str(name): gipo_teacher_diagnostics(
                    teacher,
                    split_rows,
                    context_embeddings=context_embeddings,
                    series_index_map=series_index_map,
                    schedule_grids=schedule_grids,
                    reference_time_grid=reference_time_grid,
                    density_normalizer=density_normalizer,
                    rank_temperature=float(rank_temperature),
                    regression_weight=float(regression_weight),
                    pair_margin=float(pair_margin),
                    pair_on_seed=bool(pair_on_seed),
                    split_name=str(name),
                    fit_context_ids=fit_context_ids,
                    fit_series_keys=fit_series_keys,
                    setting_feature_mode=feature_mode,
                    setting_encoder_config=encoder_config,
                    teacher_utility_weights=teacher_utility_weights,
                    complete_candidate_schedule_keys=diagnostic_candidate_schedule_map.get(str(name)),
                    device=device,
                )
                for name, split_rows in diagnostic_splits.items()
            }
            step_value = int(step + 1)
            checkpoint_history.append({"step": step_value, "diagnostics": diagnostics})
            checkpoint_states[step_value] = copy.deepcopy(teacher.state_dict())
    if bool(final_retrain_mode):
        checkpoint_selection = {
            "selection_protocol": "gipo_teacher_final_retrain",
            "selection_mode": selection_mode,
            "selection_metric": "configured_selected_step",
            "selected_step": int(steps),
            "history": checkpoint_history,
            "uses_validation_labels": False,
            "locked_test_used_for_selection": False,
        }
        selected_state = None
    else:
        checkpoint_selection, selected_state = _selected_gipo_teacher_checkpoint(
            checkpoint_history,
            checkpoint_states,
            required_split_names=tuple(diagnostic_splits.keys()) if diagnostic_splits else (),
            component_weights=teacher_selection_axis_weights,
        )
    if selected_state is not None:
        teacher.load_state_dict(selected_state)
    final_teacher_retrain = {
        "protocol": "gipo_teacher_finalization",
        "performed": False,
        "reason": "selected_checkpoint_state_restored" if selected_state is not None else "final_checkpoint_state_retained",
        "selected_checkpoint_step": checkpoint_selection.get("selected_step"),
        "selection_protocol": checkpoint_selection.get("selection_protocol"),
        "selection_metric": checkpoint_selection.get("selection_metric"),
        "uses_validation_labels": False,
        "locked_test_used_for_selection": False,
        "fit_row_count": int(len(rows)),
        "fit_context_count": int(len(fit_context_ids)),
        "fit_series_count": int(len(fit_series_keys)),
    }
    return {
        "teacher_objective": "pairwise_rank_plus_huber_regression",
        "teacher_target": "metric_vector",
        "teacher_metric_targets": list(teacher_metric_target_keys),
        "teacher_metric_target_protocol": TEACHER_METRIC_TARGET_PROTOCOL_VECTOR,
        "teacher_metric_mask_protocol": TEACHER_METRIC_MASK_PROTOCOL,
        "teacher_scalar_target": "weighted_metric_utility",
        "teacher_scalarization": TEACHER_SCALARIZATION_WEIGHTED_AVERAGE,
        "teacher_utility_weights": teacher_utility_weights_for_summary(
            teacher_metric_target_keys,
            {key: float(value) for key, value in zip(teacher_metric_target_keys, target_weights[0].detach().cpu().tolist())},
        ),
        "teacher_density_feature": "train_normalized_log_density",
        "series_conditioning": SERIES_CONDITIONING_NONE_CONTEXT_ONLY,
        "setting_feature_mode": feature_mode,
        "setting_encoder_mode": encoder_config.mode,
        "setting_encoder_config": encoder_config.to_payload(),
        "teacher_pair_count": int(left.numel()),
        "rank_temperature": float(rank_temperature),
        "regression_weight": float(regression_weight),
        "pair_margin": float(pair_margin),
        "losses": losses,
        "teacher_loss_tail_slope": _loss_tail_slope(losses, "teacher_total_loss"),
        "teacher_optimizer": {
            "optimizer": "AdamW",
            "lr": float(lr),
            "weight_decay": 1e-4,
            "steps": int(steps),
            "teacher_loss_log_every": int(teacher_loss_log_every),
            "teacher_checkpoint_every": int(checkpoint_every),
        },
        "teacher_loss_log_every": int(teacher_loss_log_every),
        "teacher_checkpoint_every": int(checkpoint_every),
        "teacher_checkpoint_selection": checkpoint_selection,
        "teacher_checkpoint_selection_mode": selection_mode,
        "final_teacher_retrain": final_teacher_retrain,
        "fit_context_count": int(len(fit_context_ids)),
        "fit_series_count": int(len(fit_series_keys)),
    }


def build_teacher_weighted_density_targets(
    teacher: nn.Module,
    rows: Sequence[MetricRow],
    *,
    context_embeddings: Mapping[str, Sequence[float]],
    series_index_map: Mapping[str, int],
    schedule_grids: Mapping[ScheduleGridKey, Sequence[float]] | None,
    reference_time_grid: Sequence[float],
    density_normalizer: DensityFeatureNormalizer,
    supervision_schedule_keys: Sequence[str] | None = None,
    temperature: float = DEFAULT_TEACHER_TARGET_TEMPERATURE,
    teacher_utility_weights: Mapping[str, float] | None = None,
    setting_feature_mode: str = SETTING_ENCODER_MODE_CONTINUOUS_V3,
    setting_encoder_config: Mapping[str, Any] | SettingEncoderConfig | None = None,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    feature_mode = validate_setting_feature_mode(setting_feature_mode)
    encoder_config = _resolve_setting_encoder_config(feature_mode, setting_encoder_config)
    fixed_temperature = _finite_positive(temperature, label="teacher_temperature")
    observed_keys = {str(row["scheduler_key"]) for row in rows}
    supervision_keys = validate_gipo_support_schedule_keys(
        sorted(observed_keys) if supervision_schedule_keys is None else supervision_schedule_keys
    )
    supervision_set = set(supervision_keys)
    validate_gipo_support_schedule_keys(sorted(observed_keys))
    unsupported_observed = sorted(observed_keys - supervision_set)
    if unsupported_observed:
        raise ValueError(f"Rows contain schedules outside supervision_schedule_keys: {unsupported_observed}")
    target_groups = _student_target_groups(rows)
    if not target_groups:
        raise ValueError("Student target construction requires at least one context group.")
    teacher_metric_target_keys = teacher_metric_target_keys_for_model(teacher)
    setting_rows: List[torch.Tensor] = []
    series_rows: List[int] = []
    context_rows: List[np.ndarray] = []
    target_masses: List[np.ndarray] = []
    entropy_values: List[float] = []
    ess_values: List[float] = []
    max_weight_values: List[float] = []
    candidate_counts: List[int] = []
    teacher.to(device)
    teacher.eval()
    score_settings: List[torch.Tensor] = []
    score_density: List[np.ndarray] = []
    score_series: List[int] = []
    score_context: List[np.ndarray] = []
    score_masses: Dict[int, Tuple[float, ...]] = {}
    for row in rows:
        context_embedding_id = context_embedding_id_from_row(row)
        if context_embedding_id not in context_embeddings:
            raise KeyError(f"Missing context embedding for {context_embedding_id}.")
        solver = str(row["solver_key"])
        target_nfe = int(row["target_nfe"])
        mass = density_mass_for_row(row, schedule_grids=schedule_grids, reference_time_grid=reference_time_grid)
        score_masses[id(row)] = mass
        score_settings.append(setting_features(solver, target_nfe, mode=feature_mode, config=encoder_config))
        score_density.append(density_normalizer.transform_one(mass, reference_time_grid=reference_time_grid))
        score_series.append(0)
        score_context.append(np.asarray(context_embeddings[context_embedding_id], dtype=np.float32))
    with torch.no_grad():
        _, metric_score_mask = _teacher_metric_targets(rows, target_keys=teacher_metric_target_keys, device=device)
        metric_score_values = _teacher_metric_scores(
            teacher,
            torch.stack(score_settings, dim=0).to(device=device),
            torch.tensor(np.stack(score_density, axis=0), dtype=torch.float32, device=device),
            torch.tensor(score_series, dtype=torch.long, device=device),
            torch.tensor(np.stack(score_context, axis=0), dtype=torch.float32, device=device),
            rows=rows,
        )
        score_weights = _teacher_metric_weights(
            rows,
            target_keys=teacher_metric_target_keys,
            batch=len(rows),
            device=metric_score_values.device,
            dtype=metric_score_values.dtype,
            teacher_utility_weights=teacher_utility_weights,
            target_mask=metric_score_mask,
        )
        score_values = _scalarize_teacher_metric_values(
            metric_score_values,
            score_weights,
            target_keys=teacher_metric_target_keys,
            target_mask=metric_score_mask,
        )
    utility_by_row_id = {
        id(row): float(value)
        for row, value in zip(rows, score_values.detach().cpu().tolist())
    }
    for _, group in target_groups:
        counts: Dict[str, int] = {key: 0 for key in supervision_keys}
        for row in group:
            key = str(row["scheduler_key"])
            counts[key] = counts.get(key, 0) + 1
        bad_counts = {key: count for key, count in counts.items() if count != 1}
        if bad_counts:
            raise ValueError(f"Teacher-weighted density targets require exactly one row per supervision schedule; counts={bad_counts}.")
        first = group[0]
        context_embedding_id = context_embedding_id_from_row(first)
        if context_embedding_id not in context_embeddings:
            raise KeyError(f"Missing context embedding for {context_embedding_id}.")
        solver = str(first["solver_key"])
        target_nfe = int(first["target_nfe"])
        masses: List[Tuple[float, ...]] = []
        utilities: List[float] = []
        setting_row = setting_features(solver, target_nfe, mode=feature_mode, config=encoder_config)
        for row in group:
            masses.append(score_masses[id(row)])
            utilities.append(float(utility_by_row_id[id(row)]))
        chosen_temperature = fixed_temperature
        utility_array = np.asarray(utilities, dtype=np.float64)
        if utility_array.size > 1:
            ordered = np.argsort(-utility_array)
        else:
            ordered = np.asarray([0], dtype=np.int64)
        weights = _teacher_candidate_weights(utilities, temperature=chosen_temperature)
        mixture = np.zeros(len(reference_time_grid) - 1, dtype=np.float64)
        for weight, mass in zip(weights, masses):
            mixture += float(weight) * np.asarray(mass, dtype=np.float64)
        mixture = mixture / max(float(np.sum(mixture)), 1e-12)
        setting_rows.append(setting_row)
        series_rows.append(0)
        context_rows.append(np.asarray(context_embeddings[context_embedding_id], dtype=np.float32))
        target_masses.append(mixture.astype(np.float32))
        entropy_values.append(float(-np.sum(weights * np.log(np.maximum(weights, 1e-12)))))
        ess_values.append(_teacher_candidate_ess(weights))
        max_weight_values.append(float(np.max(weights)))
        candidate_counts.append(int(len(group)))
    entropy_stats = _summary_percentiles(entropy_values)
    ess_stats = _summary_percentiles(ess_values)
    max_weight_stats = _summary_percentiles(max_weight_values)
    sequence_pairs = student_nfe_sequence_pairs(rows)
    summary = {
        "target_protocol": "teacher_weighted_density_mle",
        "student_target_protocol": STUDENT_TARGET_PROTOCOL_SOFT_MIXTURE,
        "teacher_output": TEACHER_OUTPUT_METRIC_VECTOR,
        "teacher_metric_targets": list(teacher_metric_target_keys),
        "teacher_metric_target_protocol": TEACHER_METRIC_TARGET_PROTOCOL_VECTOR,
        "teacher_metric_mask_protocol": TEACHER_METRIC_MASK_PROTOCOL,
        "teacher_scalarization": TEACHER_SCALARIZATION_WEIGHTED_AVERAGE,
        "teacher_utility_weights": teacher_utility_weights_for_summary(
            teacher_metric_target_keys,
            {key: float(value) for key, value in zip(teacher_metric_target_keys, score_weights[0].detach().cpu().tolist())},
        ),
        "teacher_temperature": float(fixed_temperature),
        "setting_feature_mode": feature_mode,
        "setting_encoder_mode": encoder_config.mode,
        "setting_encoder_config": encoder_config.to_payload(),
        "series_conditioning": SERIES_CONDITIONING_NONE_CONTEXT_ONLY,
        "context_setting_count": int(len(target_masses)),
        "mean_teacher_candidate_entropy": entropy_stats["mean"],
        "teacher_candidate_entropy_mean": entropy_stats["mean"],
        "teacher_candidate_entropy_p05": entropy_stats["p05"],
        "teacher_candidate_entropy_p50": entropy_stats["p50"],
        "teacher_candidate_entropy_p95": entropy_stats["p95"],
        "teacher_candidate_ess_mean": ess_stats["mean"],
        "teacher_candidate_ess_p05": ess_stats["p05"],
        "teacher_candidate_ess_p50": ess_stats["p50"],
        "teacher_candidate_ess_p95": ess_stats["p95"],
        "teacher_candidate_max_weight_mean": max_weight_stats["mean"],
        "teacher_candidate_max_weight_p05": max_weight_stats["p05"],
        "teacher_candidate_max_weight_p50": max_weight_stats["p50"],
        "teacher_candidate_max_weight_p95": max_weight_stats["p95"],
        "mean_candidate_count": float(np.mean(np.asarray(candidate_counts, dtype=np.float64))) if candidate_counts else 0.0,
        "nfe_sequence_pair_count": int(len(sequence_pairs)),
        "supervision_schedule_keys": list(supervision_keys),
        "density_protocol": DENSITY_PROTOCOL,
    }
    return (
        torch.stack(setting_rows, dim=0).to(device=device),
        torch.tensor(series_rows, dtype=torch.long, device=device),
        torch.tensor(np.stack(context_rows, axis=0), dtype=torch.float32, device=device),
        torch.tensor(np.stack(target_masses, axis=0), dtype=torch.float32, device=device),
        summary,
    )


def train_gipo_student(
    student: nn.Module,
    teacher: nn.Module,
    rows: Sequence[MetricRow],
    *,
    context_embeddings: Mapping[str, Sequence[float]],
    series_index_map: Mapping[str, int],
    schedule_grids: Mapping[ScheduleGridKey, Sequence[float]] | None,
    reference_time_grid: Sequence[float],
    density_normalizer: DensityFeatureNormalizer,
    steps: int = 500,
    lr: float = 1e-3,
    teacher_temperature: float = DEFAULT_TEACHER_TARGET_TEMPERATURE,
    teacher_utility_weights: Mapping[str, float] | None = None,
    setting_feature_mode: str = SETTING_ENCODER_MODE_CONTINUOUS_V3,
    setting_encoder_config: Mapping[str, Any] | SettingEncoderConfig | None = None,
    student_weight_decay: float = 1e-4,
    pseudo_rows: Sequence[MetricRow] | None = None,
    pseudo_context_embeddings: Mapping[str, Sequence[float]] | None = None,
    pseudo_schedule_grids: Mapping[ScheduleGridKey, Sequence[float]] | None = None,
    pseudo_target_weight: float = 0.0,
    validation_rows: Sequence[MetricRow] | None = None,
    validation_context_embeddings: Mapping[str, Sequence[float]] | None = None,
    student_log_every: int = 0,
    student_checkpoint_every: int = 100,
    final_retrain_mode: bool = False,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Student density training requires at least one fit row.")
    feature_mode = validate_setting_feature_mode(setting_feature_mode)
    encoder_config = _resolve_setting_encoder_config(feature_mode, setting_encoder_config)
    weight_decay = float(student_weight_decay)
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("student_weight_decay must be finite and nonnegative.")
    pseudo_weight = float(pseudo_target_weight)
    if not math.isfinite(pseudo_weight) or pseudo_weight < 0.0:
        raise ValueError("student_pseudo_target_weight must be finite and nonnegative.")
    selection_mode = STUDENT_CHECKPOINT_SELECTION_VALIDATION_CE
    use_validation_selection = not bool(final_retrain_mode)
    checkpoint_every = max(1, int(student_checkpoint_every))
    validation_fit_rows = [dict(row) for row in (validation_rows or [])]
    if use_validation_selection and not validation_fit_rows:
        raise ValueError("student validation CE checkpoint selection requires non-empty validation_rows.")
    locked_validation_rows = [
        row
        for row in validation_fit_rows
        if str(row.get("source_split_phase") or row.get("split_phase", row.get("split", ""))) == "locked_test"
    ]
    if locked_validation_rows:
        raise ValueError("student checkpoint selection refuses locked_test validation rows.")
    student.to(device)
    teacher.to(device)
    teacher.eval()
    sx, base_series_idx, cx, target_mass, target_summary = build_teacher_weighted_density_targets(
        teacher,
        rows,
        context_embeddings=context_embeddings,
        series_index_map=series_index_map,
        schedule_grids=schedule_grids,
        reference_time_grid=reference_time_grid,
        density_normalizer=density_normalizer,
        supervision_schedule_keys=sorted({str(row["scheduler_key"]) for row in rows}),
        temperature=float(teacher_temperature),
        teacher_utility_weights=teacher_utility_weights,
        setting_feature_mode=feature_mode,
        setting_encoder_config=encoder_config,
        device=device,
    )
    sequence_pairs = student_nfe_sequence_pairs(rows)
    target_rows = _student_target_representative_rows(rows)
    pseudo_enabled = bool(pseudo_rows) and pseudo_weight > 0.0
    pseudo_sx: torch.Tensor | None = None
    pseudo_base_series_idx: torch.Tensor | None = None
    pseudo_cx: torch.Tensor | None = None
    pseudo_target_mass: torch.Tensor | None = None
    pseudo_target_rows: List[MetricRow] = []
    pseudo_summary: Dict[str, Any] = {
        "pseudo_distillation_used": False,
        "pseudo_target_weight": float(pseudo_weight),
        "pseudo_context_setting_count": 0,
    }
    if pseudo_enabled:
        pseudo_embeddings = pseudo_context_embeddings if pseudo_context_embeddings is not None else context_embeddings
        pseudo_sx, pseudo_base_series_idx, pseudo_cx, pseudo_target_mass, built_pseudo_summary = build_teacher_weighted_density_targets(
            teacher,
            list(pseudo_rows or []),
            context_embeddings=pseudo_embeddings,
            series_index_map=series_index_map,
            schedule_grids=pseudo_schedule_grids if pseudo_schedule_grids is not None else schedule_grids,
            reference_time_grid=reference_time_grid,
            density_normalizer=density_normalizer,
            supervision_schedule_keys=sorted({str(row["scheduler_key"]) for row in (pseudo_rows or [])}),
            temperature=float(teacher_temperature),
            teacher_utility_weights=teacher_utility_weights,
            setting_feature_mode=feature_mode,
            setting_encoder_config=encoder_config,
            device=device,
        )
        pseudo_target_rows = _student_target_representative_rows(list(pseudo_rows or []))
        pseudo_summary = {
            **built_pseudo_summary,
            "pseudo_distillation_used": True,
            "pseudo_target_weight": float(pseudo_weight),
            "pseudo_context_setting_count": int(pseudo_target_mass.shape[0]),
            "pseudo_target_nfes": sorted({int(row["target_nfe"]) for row in pseudo_target_rows}),
            "pseudo_split_phases": sorted({str(row.get("source_split_phase") or row.get("split_phase", row.get("split", ""))) for row in (pseudo_rows or [])}),
        }
    validation_sx: torch.Tensor | None = None
    validation_base_series_idx: torch.Tensor | None = None
    validation_cx: torch.Tensor | None = None
    validation_target_mass: torch.Tensor | None = None
    validation_target_rows: List[MetricRow] = []
    validation_target_summary: Dict[str, Any] = {
        "student_validation_context_setting_count": 0,
        "student_validation_context_count": 0,
    }
    if validation_fit_rows:
        validation_embeddings = validation_context_embeddings if validation_context_embeddings is not None else context_embeddings
        validation_sx, validation_base_series_idx, validation_cx, validation_target_mass, built_validation_summary = build_teacher_weighted_density_targets(
            teacher,
            validation_fit_rows,
            context_embeddings=validation_embeddings,
            series_index_map=series_index_map,
            schedule_grids=schedule_grids,
            reference_time_grid=reference_time_grid,
            density_normalizer=density_normalizer,
            supervision_schedule_keys=sorted({str(row["scheduler_key"]) for row in validation_fit_rows}),
            temperature=float(teacher_temperature),
            teacher_utility_weights=teacher_utility_weights,
            setting_feature_mode=feature_mode,
            setting_encoder_config=encoder_config,
            device=device,
        )
        validation_target_rows = _student_target_representative_rows(validation_fit_rows)
        validation_target_summary = {
            **built_validation_summary,
            "student_validation_context_setting_count": int(validation_target_mass.shape[0]),
            "student_validation_context_count": int(len({context_id_from_row(row) for row in validation_fit_rows})),
            "student_validation_row_count": int(len(validation_fit_rows)),
            "student_validation_split_phases": sorted(
                {str(row.get("source_split_phase") or row.get("split_phase", row.get("split", ""))) for row in validation_fit_rows}
            ),
        }
    opt = torch.optim.AdamW(student.parameters(), lr=float(lr), weight_decay=weight_decay)
    losses: List[Dict[str, Any]] = []
    checkpoint_history: List[Dict[str, Any]] = []
    checkpoint_states: Dict[int, Mapping[str, torch.Tensor]] = {}

    def _evaluate_student_ce(
        eval_sx: torch.Tensor,
        eval_series_idx: torch.Tensor,
        eval_cx: torch.Tensor,
        eval_target_mass: torch.Tensor,
        eval_rows: Sequence[MetricRow],
    ) -> Tuple[float, float]:
        was_training = bool(student.training)
        student.eval()
        with torch.no_grad():
            eval_logits = _student_logits(
                student,
                eval_sx,
                eval_series_idx,
                eval_cx,
                rows=eval_rows,
            )
            eval_log_probs = torch.log_softmax(eval_logits, dim=-1)
            eval_ce = float((-(eval_target_mass * eval_log_probs).sum(dim=-1).mean()).detach().cpu().item())
            eval_entropy = float(
                (-(torch.softmax(eval_logits, dim=-1) * eval_log_probs).sum(dim=-1).mean()).detach().cpu().item()
            )
        if was_training:
            student.train()
        return eval_ce, eval_entropy

    student.train()
    for step in range(int(steps)):
        series_idx = base_series_idx
        logits = _student_logits(
            student,
            sx,
            series_idx,
            cx,
            rows=target_rows,
        )
        log_probs = torch.log_softmax(logits, dim=-1)
        ce_loss = -(target_mass * log_probs).sum(dim=-1).mean()
        pseudo_ce_loss = logits.new_zeros(())
        if pseudo_enabled:
            assert pseudo_sx is not None
            assert pseudo_base_series_idx is not None
            assert pseudo_cx is not None
            assert pseudo_target_mass is not None
            pseudo_series_idx = pseudo_base_series_idx
            pseudo_logits = _student_logits(
                student,
                pseudo_sx,
                pseudo_series_idx,
                pseudo_cx,
                rows=pseudo_target_rows,
            )
            pseudo_log_probs = torch.log_softmax(pseudo_logits, dim=-1)
            pseudo_ce_loss = -(pseudo_target_mass * pseudo_log_probs).sum(dim=-1).mean()
        loss = ce_loss + float(pseudo_weight) * pseudo_ce_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        should_log = _should_log_step(step, int(steps), int(student_log_every))
        should_checkpoint = use_validation_selection and _should_log_step(step, int(steps), checkpoint_every)
        train_ce_for_checkpoint: float | None = None
        train_entropy_for_checkpoint: float | None = None
        if should_log or should_checkpoint:
            train_ce_for_checkpoint, train_entropy_for_checkpoint = _evaluate_student_ce(
                sx,
                base_series_idx,
                cx,
                target_mass,
                target_rows,
            )
        if should_log:
            losses.append(
                {
                    "step": int(step + 1),
                    "student_total_loss": float(loss.detach().cpu().item()),
                    "student_kl_ce_loss": float(ce_loss.detach().cpu().item()),
                    "student_eval_kl_ce_loss": float(train_ce_for_checkpoint),
                    "student_pseudo_kl_ce_loss": float(pseudo_ce_loss.detach().cpu().item()),
                    "student_pseudo_weighted_loss": float((float(pseudo_weight) * pseudo_ce_loss).detach().cpu().item()),
                    "student_entropy": float(train_entropy_for_checkpoint),
                }
            )
        if should_checkpoint:
            assert validation_sx is not None
            assert validation_base_series_idx is not None
            assert validation_cx is not None
            assert validation_target_mass is not None
            validation_ce, validation_entropy = _evaluate_student_ce(
                validation_sx,
                validation_base_series_idx,
                validation_cx,
                validation_target_mass,
                validation_target_rows,
            )
            step_value = int(step + 1)
            checkpoint_history.append(
                {
                    "step": step_value,
                    "train_ce_loss": float(train_ce_for_checkpoint),
                    "validation_ce_loss": float(validation_ce),
                    "train_entropy": float(train_entropy_for_checkpoint),
                    "validation_entropy": float(validation_entropy),
                    "selection_constraints_passed": True,
                    "selection_constraint_failures": [],
                    "locked_test_used_for_selection": False,
                }
            )
            checkpoint_states[step_value] = copy.deepcopy(student.state_dict())
    if use_validation_selection:
        if not checkpoint_history:
            raise ValueError("Student validation CE checkpoint selection found no checkpoints.")
        selected_checkpoint = min(
            checkpoint_history,
            key=lambda entry: (
                float(entry["validation_ce_loss"]),
                float(entry["train_ce_loss"]),
                int(entry["step"]),
            ),
        )
        selected_step = int(selected_checkpoint["step"])
        selected_state = checkpoint_states.get(selected_step)
        if selected_state is not None:
            student.load_state_dict(selected_state)
        student_checkpoint_selection = {
            "selection_protocol": STUDENT_CHECKPOINT_SELECTION_VALIDATION_CE,
            "selection_mode": STUDENT_CHECKPOINT_SELECTION_VALIDATION_CE,
            "selection_metric": "validation_ce_loss",
            "selected_step": int(selected_step),
            "selected_validation_ce_loss": float(selected_checkpoint["validation_ce_loss"]),
            "selected_train_ce_loss": float(selected_checkpoint["train_ce_loss"]),
            "selected_validation_entropy": float(selected_checkpoint["validation_entropy"]),
            "selected_train_entropy": float(selected_checkpoint["train_entropy"]),
            "tie_breaker": "train_ce_then_earlier_step",
            "history": checkpoint_history,
            "student_checkpoint_every": int(checkpoint_every),
            "validation_row_count": int(len(validation_fit_rows)),
            "validation_context_count": int(len({context_id_from_row(row) for row in validation_fit_rows})),
            "uses_validation_labels": False,
            "uses_student_validation_targets": True,
            "locked_test_used_for_selection": False,
        }
    else:
        student_checkpoint_selection = {
            "selection_protocol": "gipo_student_final_retrain",
            "selection_mode": "final_retrain",
            "selection_metric": "configured_selected_step",
            "canonical_checkpoint_selection_mode": selection_mode,
            "selected_step": int(steps),
            "history": [],
            "student_checkpoint_every": int(checkpoint_every),
            "uses_validation_labels": False,
            "uses_student_validation_targets": False,
            "locked_test_used_for_selection": False,
        }
    return {
        "student_policy_type": "continuous_density",
        "student_objective": "teacher_weighted_density_mle_kl",
        "student_target_protocol": STUDENT_TARGET_PROTOCOL_SOFT_MIXTURE,
        "density_protocol": DENSITY_PROTOCOL,
        "student_target_summary": target_summary,
        "student_validation_target_summary": validation_target_summary,
        "student_pseudo_target_summary": pseudo_summary,
        "teacher_utility_weights": dict(target_summary.get("teacher_utility_weights", {})),
        "pseudo_distillation_used": bool(pseudo_enabled),
        "pseudo_target_weight": float(pseudo_weight),
        "series_conditioning": SERIES_CONDITIONING_NONE_CONTEXT_ONLY,
        "student_weight_decay": float(weight_decay),
        "setting_feature_mode": feature_mode,
        "setting_encoder_mode": encoder_config.mode,
        "setting_encoder_config": encoder_config.to_payload(),
        "student_nfe_sequence_pair_count": int(len(sequence_pairs)),
        "losses": losses,
        "student_loss_tail_slope": _loss_tail_slope(losses, "student_kl_ce_loss"),
        "student_eval_loss_tail_slope": _loss_tail_slope(losses, "student_eval_kl_ce_loss"),
        "student_optimizer": {
            "optimizer": "AdamW",
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "steps": int(steps),
            "student_log_every": int(student_log_every),
            "student_checkpoint_every": int(checkpoint_every),
            "student_checkpoint_selection_mode": selection_mode,
        },
        "student_log_every": int(student_log_every),
        "student_checkpoint_every": int(checkpoint_every),
        "student_checkpoint_selection_mode": selection_mode,
        "student_checkpoint_selection": student_checkpoint_selection,
        "student_validation_split": {
            "protocol": "context_disjoint_student_validation",
            "validation_row_count": int(len(validation_fit_rows)),
            "validation_context_count": int(len({context_id_from_row(row) for row in validation_fit_rows})) if validation_fit_rows else 0,
            "locked_test_used_for_selection": False,
        },
        "student_validation_used_for_selection": bool(use_validation_selection),
        "locked_test_used_for_selection": False,
        "student_final_retrain": {
            "enabled": bool(final_retrain_mode),
            "performed": bool(final_retrain_mode),
            "selected_step": int(student_checkpoint_selection.get("selected_step") or int(steps)),
            "selection_protocol": str(student_checkpoint_selection.get("selection_protocol", selection_mode)),
            "locked_test_used_for_selection": False,
        },
    }


def predict_gipo_density(
    student: nn.Module,
    *,
    row: MetricRow,
    context_embedding: Sequence[float],
    series_index_map: Mapping[str, int],
    reference_time_grid: Sequence[float],
    setting_feature_mode: str = SETTING_ENCODER_MODE_CONTINUOUS_V3,
    setting_encoder_config: Mapping[str, Any] | SettingEncoderConfig | None = None,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    return predict_gipo_density_many(
        student,
        rows=[row],
        context_embeddings={context_embedding_id_from_row(row): context_embedding},
        series_index_map=series_index_map,
        reference_time_grid=reference_time_grid,
        setting_feature_mode=setting_feature_mode,
        setting_encoder_config=setting_encoder_config,
        device=device,
    )[0]


def predict_gipo_density_many(
    student: nn.Module,
    *,
    rows: Sequence[MetricRow],
    context_embeddings: Mapping[str, Sequence[float]],
    series_index_map: Mapping[str, int],
    reference_time_grid: Sequence[float],
    setting_feature_mode: str = SETTING_ENCODER_MODE_CONTINUOUS_V3,
    setting_encoder_config: Mapping[str, Any] | SettingEncoderConfig | None = None,
    device: torch.device | str = "cpu",
) -> List[Dict[str, Any]]:
    if not rows:
        return []
    feature_mode = validate_setting_feature_mode(setting_feature_mode)
    encoder_config = _resolve_setting_encoder_config(feature_mode, setting_encoder_config)
    setting_rows: List[torch.Tensor] = []
    series_rows: List[int] = []
    context_rows: List[np.ndarray] = []
    for row in rows:
        context_embedding_id = context_embedding_id_from_row(row)
        if context_embedding_id not in context_embeddings:
            raise KeyError(f"Missing context embedding for {context_embedding_id}.")
        setting_rows.append(setting_features(str(row["solver_key"]), int(row["target_nfe"]), mode=feature_mode, config=encoder_config))
        series_rows.append(0)
        context_rows.append(np.asarray(context_embeddings[context_embedding_id], dtype=np.float32))
    student.to(device)
    student.eval()
    with torch.no_grad():
        masses_t = _student_density_mass(
            student,
            torch.stack(setting_rows, dim=0).to(device=device),
            torch.tensor(series_rows, dtype=torch.long, device=device),
            torch.tensor(np.stack(context_rows, axis=0), dtype=torch.float32, device=device),
            rows=rows,
        )
    outputs: List[Dict[str, Any]] = []
    for row, mass_t in zip(rows, masses_t):
        solver = str(row["solver_key"])
        target_nfe = int(row["target_nfe"])
        macro_steps = solver_macro_steps(solver, target_nfe)
        mass = tuple(float(x) for x in mass_t.detach().cpu().numpy().astype(np.float64).tolist())
        grid = density_mass_to_time_grid(mass, macro_steps=macro_steps, reference_time_grid=reference_time_grid)
        outputs.append(
            {
                "solver_key": solver,
                "target_nfe": int(target_nfe),
                "macro_steps": int(macro_steps),
                "time_grid": list(grid),
                "schedule_grid_hash": schedule_grid_hash(grid),
                "density_mass": [float(x) for x in mass],
                "density_mass_hash": density_mass_hash(mass, reference_time_grid=reference_time_grid),
                "setting_feature_mode": feature_mode,
                "setting_encoder_mode": encoder_config.mode,
                "setting_encoder_config": encoder_config.to_payload(),
                **density_metadata(reference_time_grid),
            }
        )
    return outputs


def read_metric_rows_csv(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as fh:
        rows = [dict(row) for row in csv.DictReader(fh)]
    for row in rows:
        if str(row.get("solver_key", "")).strip():
            row["solver_key"] = normalize_solver_key(str(row["solver_key"]))
    return rows


__all__ = [
    "GIPO_PROTOCOL",
    "GIPO_SUPPORT_SCHEDULE_KEYS",
    "DEFAULT_CONTEXT_CALIBRATION_TOTAL",
    "DEFAULT_DENSITY_BIN_COUNT",
    "DEFAULT_SUPPORT_SCHEDULE_KEYS",
    "DEFAULT_SUPERVISION_SCHEDULE_KEYS",
    "EXPERIMENTAL_SUPERVISION_SCHEDULE_KEYS",
    "DEFAULT_TEACHER_TARGET_TEMPERATURE",
    "STUDENT_TARGET_PROTOCOL_SOFT_MIXTURE",
    "DEFAULT_DENSITY_FAMILY_HOLDOUT_SCHEDULE_KEYS",
    "DEFAULT_TEACHER_CHECKPOINT_SELECTION_MODE",
    "DEFAULT_STUDENT_CHECKPOINT_SELECTION_MODE",
    "DEFAULT_TEACHER_SELECTION_COMPONENT_WEIGHTS",
    "ARCHITECTURE_DENSITY_FORM_TRANSFORMER",
    "ARCHITECTURE_DENSITY_QUERY_TRANSFORMER",
    "CONDITIONING_STYLE_ADDITIVE_MLP",
    "DENSITY_TOKEN_ATTENTION_ROPE",
    "DEFAULT_TRANSFORMER_DROPOUT",
    "DEFAULT_TRANSFORMER_HEADS",
    "DEFAULT_TRANSFORMER_HIDDEN_DIM",
    "DEFAULT_TRANSFORMER_LAYERS",
    "MODEL_PAYLOAD_VERSION",
    "MAX_CONTEXT_CALIBRATION_TOTAL",
    "MIN_CONTEXT_CALIBRATION_TOTAL",
    "SERIES_CONDITIONING_DIM",
    "SERIES_CONDITIONING_NONE_CONTEXT_ONLY",
    "STUDENT_CHECKPOINT_SELECTION_VALIDATION_CE",
    "TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET",
    "TEACHER_METRIC_MASK_PROTOCOL",
    "TEACHER_METRIC_TARGET_PROTOCOL_VECTOR",
    "GIPODensityFormTeacherTransformer",
    "GIPODensityQueryStudentTransformer",
    "DensityFeatureNormalizer",
    "EmbeddingNormalizer",
    "attach_uniform_gipo_rewards",
    "build_series_index_map",
    "build_gipo_student_model",
    "build_gipo_teacher_model",
    "build_teacher_weighted_density_targets",
    "context_calibration_train_val_counts",
    "context_embedding_id_from_row",
    "context_id_from_row",
    "context_pair_key",
    "density_family_for_schedule_key",
    "gipo_teacher_diagnostics",
    "density_mass_for_row",
    "grid_for_schedule",
    "load_context_embedding_table",
    "nfe_sequence_diagnostic_summary",
    "pairwise_rank_loss",
    "predict_gipo_density",
    "predict_gipo_density_many",
    "read_metric_rows_csv",
    "recommended_context_calibration_count",
    "remapped_series_index",
    "sample_context_ids_stratified",
    "save_context_embedding_table",
    "schedule_grid_hash",
    "select_weighted_normalized_regret_checkpoint",
    "series_key_from_row",
    "logical_seed_from_row",
    "evaluation_seed_from_row",
    "realized_nfe_from_row",
    "teacher_selection_candidate_group_key",
    "split_rows_by_context_holdout",
    "split_rows_by_density_family_holdout",
    "student_nfe_sequence_pair_indices",
    "student_nfe_sequence_pairs",
    "train_gipo_student",
    "train_gipo_teacher",
    "validate_gipo_architecture",
    "validate_gipo_teacher_training_metadata",
    "validate_gipo_support_schedule_keys",
    "validate_density_family_holdout_schedule_keys",
    "validate_series_conditioning",
    "TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET",
    "validate_teacher_objective_hyperparameters",
    "validate_reference_grid",
]
