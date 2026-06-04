from __future__ import annotations

import copy
import csv
import hashlib
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

from genode.gipo.density_representation import (
    DEFAULT_DENSITY_BIN_COUNT,
    DENSITY_PROTOCOL,
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
    SETTING_FEATURE_MODE_GIPO_V1,
    setting_features,
    solver_macro_steps,
    validate_setting_feature_mode,
    validate_time_grid,
)
from genode.gipo.objectives import DEFAULT_REWARD_EPS, UNIFORM_SCHEDULE_KEY
from genode.gipo.ser_ptg_reference import SER_PTG_SCHEDULE_KEY
from genode.schedule_transfer.diffusion_flow_schedules import (
    BASELINE_SCHEDULE_KEYS,
    EXPERIMENTAL_FIXED_SCHEDULE_KEYS,
    build_schedule_grid,
)

MetricRow = Mapping[str, Any]
ContextPairKey = Tuple[str, str, int, str, int | None]
ScheduleGridKey = Tuple[str, str, int]

GIPO_PROTOCOL = "gipo_density_v1"
DEFAULT_SUPERVISION_SCHEDULE_KEYS: Tuple[str, ...] = tuple(BASELINE_SCHEDULE_KEYS) + (SER_PTG_SCHEDULE_KEY,)
DEFAULT_SUPPORT_SCHEDULE_KEYS: Tuple[str, ...] = DEFAULT_SUPERVISION_SCHEDULE_KEYS
GIPO_SUPPORT_SCHEDULE_KEYS: Tuple[str, ...] = DEFAULT_SUPERVISION_SCHEDULE_KEYS
EXPERIMENTAL_SUPERVISION_SCHEDULE_KEYS: Tuple[str, ...] = tuple(EXPERIMENTAL_FIXED_SCHEDULE_KEYS) + (SER_PTG_SCHEDULE_KEY,)
DEFAULT_CONTEXT_CALIBRATION_TOTAL = 120
DEFAULT_CONTEXT_CALIBRATION_VALIDATION_FRACTION = 0.20
MIN_CONTEXT_CALIBRATION_TOTAL = 72
MAX_CONTEXT_CALIBRATION_TOTAL = 144
DEFAULT_TEACHER_TARGET_TEMPERATURE = 0.05
TEACHER_TEMPERATURE_MODE_FIXED = "fixed"
TEACHER_TEMPERATURE_MODE_ADAPTIVE_ESS = "adaptive_ess"
DEFAULT_TEACHER_TARGET_ESS = 2.5
DEFAULT_TEACHER_MIN_TEMPERATURE = 0.01
DEFAULT_TEACHER_MAX_TEMPERATURE = 1.0
STUDENT_TARGET_MODE_SOFT_MIXTURE = "soft_mixture"
STUDENT_TARGET_MODE_MARGIN_HARD_SOFT = "margin_hard_soft"
DEFAULT_TEACHER_HARD_MARGIN = 0.05


def _normalized_metric_weights(crps_weight: float, mase_weight: float) -> Tuple[float, float]:
    crps = float(crps_weight)
    mase = float(mase_weight)
    if not math.isfinite(crps) or not math.isfinite(mase) or crps < 0.0 or mase < 0.0:
        raise ValueError("teacher utility weights must be finite and nonnegative.")
    total = crps + mase
    if total <= 0.0:
        raise ValueError("At least one teacher utility metric weight must be positive.")
    return float(crps / total), float(mase / total)


def _finite_positive(value: Any, *, label: str) -> float:
    val = float(value)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"{label} must be finite and positive, got {value!r}.")
    return val


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


def _teacher_temperature_for_target_ess(
    utilities: Sequence[float],
    *,
    target_ess: float,
    min_temperature: float,
    max_temperature: float,
) -> float:
    min_temp = _finite_positive(min_temperature, label="teacher_min_temperature")
    max_temp = _finite_positive(max_temperature, label="teacher_max_temperature")
    if min_temp > max_temp:
        raise ValueError("teacher_min_temperature must be <= teacher_max_temperature.")
    util_arr = np.asarray(utilities, dtype=np.float64)
    if util_arr.size == 0:
        raise ValueError("Adaptive ESS temperature requires at least one teacher utility.")
    bounded_target = min(float(util_arr.size), max(1.0, _finite_positive(target_ess, label="teacher_target_ess")))
    low_ess = _teacher_candidate_ess(_teacher_candidate_weights(util_arr, temperature=min_temp))
    high_ess = _teacher_candidate_ess(_teacher_candidate_weights(util_arr, temperature=max_temp))
    if bounded_target <= low_ess:
        return float(min_temp)
    if bounded_target >= high_ess:
        return float(max_temp)
    low = float(min_temp)
    high = float(max_temp)
    for _ in range(64):
        mid = 0.5 * (low + high)
        mid_ess = _teacher_candidate_ess(_teacher_candidate_weights(util_arr, temperature=mid))
        if mid_ess < bounded_target:
            low = mid
        else:
            high = mid
    return float(0.5 * (low + high))


def validate_teacher_temperature_mode(mode: str) -> str:
    value = str(mode).strip()
    allowed = {TEACHER_TEMPERATURE_MODE_FIXED, TEACHER_TEMPERATURE_MODE_ADAPTIVE_ESS}
    if value not in allowed:
        raise ValueError(f"teacher_temperature_mode must be one of {sorted(allowed)}, got {mode!r}.")
    return value


def validate_student_target_mode(mode: str) -> str:
    value = str(mode).strip() or STUDENT_TARGET_MODE_SOFT_MIXTURE
    allowed = {STUDENT_TARGET_MODE_SOFT_MIXTURE, STUDENT_TARGET_MODE_MARGIN_HARD_SOFT}
    if value not in allowed:
        raise ValueError(f"student_target_mode must be one of {sorted(allowed)}, got {mode!r}.")
    return value


def _json_hash(payload: Mapping[str, Any], *, prefix: str) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:24]}"


def schedule_grid_hash(time_grid: Sequence[float]) -> str:
    values = [round(float(value), 12) for value in time_grid]
    return _json_hash({"time_grid": values}, prefix="grid")


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
) -> str:
    return _json_hash(
        {
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
    )


def series_key_from_row(row: MetricRow) -> str:
    series_id = str(row.get("series_id", "") or "").strip()
    if series_id:
        return series_id
    series_idx = str(row.get("series_idx", "") or "").strip()
    if series_idx:
        return f"series_idx:{series_idx}"
    raise ValueError("Rows require series_id or series_idx for series-disjoint diagnostics.")


def context_pair_key(row: MetricRow, *, pair_on_seed: bool = True) -> ContextPairKey:
    seed = _optional_int(row.get("seed")) if pair_on_seed else None
    return (
        str(row.get("dataset", row.get("dataset_key", ""))),
        str(row["solver_key"]),
        int(row["target_nfe"]),
        context_id_from_row(row),
        seed,
    )


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
        uniform_crps = _finite_positive(uniform_rows[0]["crps"], label="uniform_crps")
        uniform_mase = _finite_positive(uniform_rows[0]["mase"], label="uniform_mase")
        e = float(eps)
        for row in group:
            crps = _finite_positive(row["crps"], label="crps")
            mase = _finite_positive(row["mase"], label="mase")
            u_crps = float(-math.log((crps + e) / (uniform_crps + e)))
            u_mase = float(-math.log((mase + e) / (uniform_mase + e)))
            copied = dict(row)
            copied.update(
                {
                    "gipo_reward_protocol": GIPO_PROTOCOL,
                    "reward_anchor_schedule_key": uniform_key,
                    "uniform_crps": float(uniform_crps),
                    "uniform_mase": float(uniform_mase),
                    "u_crps_uniform": u_crps,
                    "u_mase_uniform": u_mase,
                    "u_comp_crps_weight": float(crps_weight),
                    "u_comp_mase_weight": float(mase_weight),
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


def split_rows_by_series_holdout(
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
    series_keys = sorted({series_key_from_row(row) for row in rows})
    if len(series_keys) <= 1:
        return [dict(row) for row in rows], []
    rng = np.random.default_rng(int(seed))
    shuffled = list(series_keys)
    rng.shuffle(shuffled)
    holdout_count = max(1, int(round(float(len(shuffled)) * frac)))
    holdout_count = min(holdout_count, len(shuffled) - 1)
    holdout_series = set(shuffled[:holdout_count])
    fit_rows: List[Dict[str, Any]] = []
    holdout_rows: List[Dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["context_id"] = context_id_from_row(copied)
        copied["series_key"] = series_key_from_row(copied)
        if copied["series_key"] in holdout_series:
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
    cap = min(int(max_total), max(int(min_total), 12 * int(cells)))
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
    strata: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    for context_id, row in by_context.items():
        series = str(row.get("series_id", row.get("series_idx", "")))
        try:
            target_t = int(row.get("target_t", 0))
        except (TypeError, ValueError):
            target_t = 0
        strata[(series, int(math.floor(float(target_t) / 24.0)))].append(context_id)
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
        "path": str(resolved),
        "context_count": int(context_ids.size),
        "embedding_dim": int(matrix.shape[1]),
        "metadata": dict(metadata or {}),
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


def grid_for_schedule(
    schedule_key: str,
    solver_key: str,
    target_nfe: int,
    *,
    schedule_grids: Mapping[ScheduleGridKey, Sequence[float]] | None = None,
) -> Tuple[float, ...]:
    key = str(schedule_key)
    macro_steps = solver_macro_steps(str(solver_key), int(target_nfe))
    if schedule_grids is not None and (key, str(solver_key), int(target_nfe)) in schedule_grids:
        return validate_time_grid(schedule_grids[(key, str(solver_key), int(target_nfe))], macro_steps=macro_steps)
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
    grid = grid_for_schedule(str(row["scheduler_key"]), solver, target_nfe, schedule_grids=schedule_grids)
    return grid_to_density_mass(grid, reference_time_grid=reference_time_grid, macro_steps=solver_macro_steps(solver, target_nfe))


class GIPOScheduleTeacherMLP(nn.Module):
    """Teacher scorer for `(solver, nfe, density, series, context)`."""

    def __init__(
        self,
        *,
        setting_dim: int,
        density_dim: int,
        context_dim: int,
        num_series: int,
        series_embedding_dim: int = 32,
        hidden_dim: int = 256,
        hidden_layers: int = 3,
    ):
        super().__init__()
        self.setting_dim = int(setting_dim)
        self.density_dim = int(density_dim)
        self.context_dim = int(context_dim)
        self.num_series = int(num_series)
        self.unknown_series_index = int(num_series)
        self.series_embedding = nn.Embedding(int(num_series) + 1, int(series_embedding_dim))
        input_dim = int(setting_dim) + int(density_dim) + int(series_embedding_dim) + int(context_dim)
        layers: List[nn.Module] = []
        dim = input_dim
        for _ in range(int(hidden_layers)):
            layers.extend([nn.Linear(dim, int(hidden_dim)), nn.SiLU()])
            dim = int(hidden_dim)
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        setting_feature_batch: torch.Tensor,
        density_feature_batch: torch.Tensor,
        series_index_batch: torch.Tensor,
        context_embedding_batch: torch.Tensor,
    ) -> torch.Tensor:
        series_index_batch = series_index_batch.reshape(-1)
        batch = int(setting_feature_batch.shape[0])
        if setting_feature_batch.ndim != 2 or setting_feature_batch.shape[-1] != self.setting_dim:
            raise ValueError("setting_feature_batch must be 2D with setting_dim columns.")
        if density_feature_batch.ndim != 2 or density_feature_batch.shape[-1] != self.density_dim:
            raise ValueError("density_feature_batch must be 2D with density_dim columns.")
        if context_embedding_batch.ndim != 2 or context_embedding_batch.shape[-1] != self.context_dim:
            raise ValueError("context_embedding_batch must be 2D with context_dim columns.")
        if density_feature_batch.shape[0] != batch or context_embedding_batch.shape[0] != batch or series_index_batch.shape[0] != batch:
            raise ValueError("Teacher feature batches must share the same batch dimension.")
        series_idx = torch.clamp(series_index_batch.to(dtype=torch.long, device=setting_feature_batch.device), min=0, max=self.unknown_series_index)
        series_emb = self.series_embedding(series_idx)
        features = torch.cat(
            [
                setting_feature_batch,
                density_feature_batch.to(device=setting_feature_batch.device, dtype=setting_feature_batch.dtype),
                series_emb.to(dtype=setting_feature_batch.dtype),
                context_embedding_batch.to(device=setting_feature_batch.device, dtype=setting_feature_batch.dtype),
            ],
            dim=-1,
        )
        return self.net(features).squeeze(-1)


class GIPODensityStudentMLP(nn.Module):
    """Context-conditioned continuous density policy over normalized solver time."""

    def __init__(
        self,
        *,
        setting_dim: int,
        density_dim: int,
        context_dim: int,
        num_series: int,
        series_embedding_dim: int = 32,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
    ):
        super().__init__()
        self.setting_dim = int(setting_dim)
        self.density_dim = int(density_dim)
        self.context_dim = int(context_dim)
        self.num_series = int(num_series)
        self.unknown_series_index = int(num_series)
        self.series_embedding = nn.Embedding(int(num_series) + 1, int(series_embedding_dim))
        input_dim = int(setting_dim) + int(series_embedding_dim) + int(context_dim)
        layers: List[nn.Module] = []
        dim = input_dim
        for _ in range(int(hidden_layers)):
            layers.extend([nn.Linear(dim, int(hidden_dim)), nn.SiLU()])
            dim = int(hidden_dim)
        layers.append(nn.Linear(dim, int(density_dim)))
        self.net = nn.Sequential(*layers)

    def logits(
        self,
        setting_feature_batch: torch.Tensor,
        series_index_batch: torch.Tensor,
        context_embedding_batch: torch.Tensor,
    ) -> torch.Tensor:
        series_index_batch = series_index_batch.reshape(-1)
        batch = int(setting_feature_batch.shape[0])
        if setting_feature_batch.ndim != 2 or setting_feature_batch.shape[-1] != self.setting_dim:
            raise ValueError("setting_feature_batch must be 2D with setting_dim columns.")
        if context_embedding_batch.ndim != 2 or context_embedding_batch.shape[-1] != self.context_dim:
            raise ValueError("context_embedding_batch must be 2D with context_dim columns.")
        if context_embedding_batch.shape[0] != batch or series_index_batch.shape[0] != batch:
            raise ValueError("Student feature batches must share the same batch dimension.")
        series_idx = torch.clamp(series_index_batch.to(dtype=torch.long, device=setting_feature_batch.device), min=0, max=self.unknown_series_index)
        series_emb = self.series_embedding(series_idx)
        features = torch.cat(
            [
                setting_feature_batch,
                series_emb.to(dtype=setting_feature_batch.dtype),
                context_embedding_batch.to(device=setting_feature_batch.device, dtype=setting_feature_batch.dtype),
            ],
            dim=-1,
        )
        return self.net(features)

    def density_mass(
        self,
        setting_feature_batch: torch.Tensor,
        series_index_batch: torch.Tensor,
        context_embedding_batch: torch.Tensor,
    ) -> torch.Tensor:
        return torch.softmax(self.logits(setting_feature_batch, series_index_batch, context_embedding_batch), dim=-1)


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
    temp = max(float(temperature), 1e-6)
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
    setting_feature_mode: str = SETTING_FEATURE_MODE_GIPO_V1,
    series_unknown_probability: float = 0.0,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[ContextPairKey], List[str], List[Tuple[float, ...]]]:
    feature_mode = validate_setting_feature_mode(setting_feature_mode)
    setting_rows: List[torch.Tensor] = []
    density_rows: List[np.ndarray] = []
    series_rows: List[int] = []
    context_rows: List[np.ndarray] = []
    targets: List[float] = []
    pair_keys: List[ContextPairKey] = []
    schedule_keys: List[str] = []
    density_masses: List[Tuple[float, ...]] = []
    rng = np.random.default_rng(int(seed))
    unknown_index = int(len(series_index_map))
    unknown_probability = min(1.0, max(0.0, float(series_unknown_probability)))
    for row in rows:
        context_id = context_id_from_row(row)
        if context_id not in context_embeddings:
            raise KeyError(f"Missing context embedding for {context_id}.")
        solver = str(row["solver_key"])
        target_nfe = int(row["target_nfe"])
        mass = density_mass_for_row(row, schedule_grids=schedule_grids, reference_time_grid=reference_time_grid)
        setting_rows.append(setting_features(solver, target_nfe, mode=feature_mode))
        density_rows.append(density_normalizer.transform_one(mass, reference_time_grid=reference_time_grid))
        series_idx = remapped_series_index(row, series_index_map)
        if unknown_probability > 0.0 and rng.random() < unknown_probability:
            series_idx = unknown_index
        series_rows.append(series_idx)
        context_rows.append(np.asarray(context_embeddings[context_id], dtype=np.float32))
        targets.append(float(row["u_comp_uniform"]))
        pair_keys.append(context_pair_key(row, pair_on_seed=True))
        schedule_keys.append(str(row["scheduler_key"]))
        density_masses.append(mass)
    return (
        torch.stack(setting_rows, dim=0).to(device=device),
        torch.tensor(np.stack(density_rows, axis=0), dtype=torch.float32, device=device),
        torch.tensor(series_rows, dtype=torch.long, device=device),
        torch.tensor(np.stack(context_rows, axis=0), dtype=torch.float32, device=device),
        torch.tensor(targets, dtype=torch.float32, device=device),
        pair_keys,
        schedule_keys,
        density_masses,
    )


def gipo_teacher_diagnostics(
    teacher: GIPOScheduleTeacherMLP,
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
    setting_feature_mode: str = SETTING_FEATURE_MODE_GIPO_V1,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    feature_mode = validate_setting_feature_mode(setting_feature_mode)
    contexts = {context_id_from_row(row) for row in rows}
    series_keys = {series_key_from_row(row) for row in rows}
    schedules = {str(row["scheduler_key"]) for row in rows}
    fit_context_set = {str(value) for value in fit_context_ids}
    fit_series_set = {str(value) for value in fit_series_keys}
    base: Dict[str, Any] = {
        "split_name": str(split_name),
        "row_count": int(len(rows)),
        "context_count": int(len(contexts)),
        "series_count": int(len(series_keys)),
        "schedule_count": int(len(schedules)),
        "schedule_keys": sorted(schedules),
        "split_phases": sorted({str(row.get("split_phase", row.get("split", ""))) for row in rows}),
        "fit_context_overlap_count": int(len(contexts & fit_context_set)),
        "fit_series_overlap_count": int(len(series_keys & fit_series_set)),
        "setting_feature_mode": feature_mode,
        "uses_validation_labels": False,
    }
    if not rows:
        base.update({"rank_loss": None, "huber_loss": None, "total_loss": None, "pairwise_accuracy": None, "pair_count": 0, "spearman_rank_correlation": None})
        return base
    was_training = bool(teacher.training)
    teacher.eval()
    with torch.no_grad():
        sx, dx, series_idx, cx, targets, pair_keys, schedule_keys, _ = _teacher_training_tensors(
            rows,
            context_embeddings=context_embeddings,
            series_index_map=series_index_map,
            schedule_grids=schedule_grids,
            reference_time_grid=reference_time_grid,
            density_normalizer=density_normalizer,
            setting_feature_mode=feature_mode,
            device=device,
        )
        if not pair_on_seed:
            pair_keys = [key[:4] + (None,) for key in pair_keys]
        left, right, sign = _pair_indices(targets, pair_keys, margin=float(pair_margin), device=sx.device)
        pred = teacher(sx, dx, series_idx, cx)
        rank = pairwise_rank_loss(pred, left, right, sign, temperature=float(rank_temperature))
        huber = F.smooth_l1_loss(pred, targets)
        total = rank + float(regression_weight) * huber
        if left.numel() > 0:
            predicted_sign = torch.sign(pred[left] - pred[right])
            pairwise_accuracy = float((predicted_sign == sign).to(torch.float32).mean().detach().cpu().item())
        else:
            pairwise_accuracy = None
        pred_values = [float(value) for value in pred.detach().cpu().tolist()]
        target_values = [float(value) for value in targets.detach().cpu().tolist()]
    if was_training:
        teacher.train()

    by_group_schedule: Dict[Tuple[str, str, int, str], Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for idx, key in enumerate(pair_keys):
        by_group_schedule[(str(key[0]), str(key[1]), int(key[2]), str(key[3]))][str(schedule_keys[idx])].append(int(idx))
    spearman_values: List[float] = []
    best_candidate_hits = 0
    best_candidate_total = 0
    for schedule_to_indices in by_group_schedule.values():
        if len(schedule_to_indices) < 2:
            continue
        pred_by_schedule = {
            schedule: float(np.mean(np.asarray([pred_values[idx] for idx in indices], dtype=np.float64)))
            for schedule, indices in schedule_to_indices.items()
        }
        target_by_schedule = {
            schedule: float(np.mean(np.asarray([target_values[idx] for idx in indices], dtype=np.float64)))
            for schedule, indices in schedule_to_indices.items()
        }
        spearman_values.append(_spearman_for_scores(pred_by_schedule, target_by_schedule))
        best_candidate_hits += int(
            sorted(pred_by_schedule, key=lambda schedule: (-pred_by_schedule[schedule], schedule))[0]
            == sorted(target_by_schedule, key=lambda schedule: (-target_by_schedule[schedule], schedule))[0]
        )
        best_candidate_total += 1
    base.update(
        {
            "rank_loss": float(rank.detach().cpu().item()),
            "huber_loss": float(huber.detach().cpu().item()),
            "total_loss": float(total.detach().cpu().item()),
            "pairwise_accuracy": pairwise_accuracy,
            "pair_count": int(left.numel()),
            "spearman_rank_correlation": float(np.mean(np.asarray(spearman_values, dtype=np.float64))) if spearman_values else None,
            "best_candidate_agreement": None if best_candidate_total == 0 else float(best_candidate_hits / best_candidate_total),
            "candidate_group_count": int(best_candidate_total),
        }
    )
    return base


def _selected_gipo_teacher_checkpoint(
    checkpoint_history: Sequence[Dict[str, Any]],
    checkpoint_states: Mapping[int, Mapping[str, torch.Tensor]],
    *,
    required_split_names: Sequence[str] = (),
) -> Tuple[Dict[str, Any], Mapping[str, torch.Tensor] | None]:
    if not checkpoint_history:
        return (
            {
                "selection_protocol": "final_checkpoint_no_diagnostics",
                "selection_metric": "final_step",
                "selected_step": None,
                "history": [],
                "uses_validation_labels": False,
            },
            None,
        )
    required = tuple(str(name) for name in required_split_names)
    scored: List[Dict[str, Any]] = []
    best_entry: Dict[str, Any] | None = None
    best_key: Tuple[float, float, float, int] | None = None
    for entry in checkpoint_history:
        diagnostics = {str(key): dict(value) for key, value in dict(entry.get("diagnostics", {})).items()}
        missing = [name for name in required if diagnostics.get(name, {}).get("total_loss") is None]
        if missing:
            copied = dict(entry)
            copied["selection_constraints_passed"] = False
            copied["selection_constraint_failures"] = [f"{name}:missing_total_loss" for name in missing]
            scored.append(copied)
            continue
        active_names = required or tuple(sorted(name for name, diag in diagnostics.items() if diag.get("total_loss") is not None))
        losses = [float(diagnostics[name]["total_loss"]) for name in active_names]
        pairwise_values = [float(diagnostics[name]["pairwise_accuracy"]) for name in active_names if diagnostics[name].get("pairwise_accuracy") is not None]
        spearman_values = [float(diagnostics[name]["spearman_rank_correlation"]) for name in active_names if diagnostics[name].get("spearman_rank_correlation") is not None]
        copied = dict(entry)
        copied["diagnostics"] = diagnostics
        copied["selection_constraints_passed"] = True
        copied["selection_constraint_failures"] = []
        copied["mean_diagnostic_total_loss"] = float(np.mean(np.asarray(losses, dtype=np.float64)))
        copied["mean_pairwise_accuracy"] = float(np.mean(np.asarray(pairwise_values, dtype=np.float64))) if pairwise_values else 0.0
        copied["mean_spearman_rank_correlation"] = float(np.mean(np.asarray(spearman_values, dtype=np.float64))) if spearman_values else 0.0
        scored.append(copied)
        key = (
            float(copied["mean_diagnostic_total_loss"]),
            -float(copied["mean_pairwise_accuracy"]),
            -float(copied["mean_spearman_rank_correlation"]),
            int(copied["step"]),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_entry = copied
    if best_entry is None:
        raise ValueError("Teacher checkpoint selection found no checkpoint with all required diagnostic split losses.")
    selected_step = int(best_entry["step"])
    return (
        {
            "selection_protocol": "gipo_teacher_checkpoint",
            "selection_split": "context_and_series_teacher_holdouts",
            "selection_metric": "mean_context_series_total_loss",
            "selected_step": selected_step,
            "selected_mean_diagnostic_total_loss": best_entry.get("mean_diagnostic_total_loss"),
            "tie_breaker": "higher_pairwise_then_higher_spearman_then_earlier_step",
            "uses_validation_labels": False,
            "history": scored,
        },
        checkpoint_states.get(selected_step),
    )


def train_gipo_teacher(
    teacher: GIPOScheduleTeacherMLP,
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
    teacher_checkpoint_every: int = 100,
    series_unknown_probability: float = 0.0,
    seed: int = 0,
    allowed_schedule_keys: Sequence[str] = DEFAULT_SUPERVISION_SCHEDULE_KEYS,
    setting_feature_mode: str = SETTING_FEATURE_MODE_GIPO_V1,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Teacher training requires at least one context reward row.")
    validate_gipo_support_schedule_keys(sorted({str(row["scheduler_key"]) for row in rows}), allowed_schedule_keys=allowed_schedule_keys)
    feature_mode = validate_setting_feature_mode(setting_feature_mode)
    teacher.to(device)
    sx, dx, series_idx, cx, targets, pair_keys, _, _ = _teacher_training_tensors(
        rows,
        context_embeddings=context_embeddings,
        series_index_map=series_index_map,
        schedule_grids=schedule_grids,
        reference_time_grid=reference_time_grid,
        density_normalizer=density_normalizer,
        device=device,
        setting_feature_mode=feature_mode,
        series_unknown_probability=float(series_unknown_probability),
        seed=int(seed),
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
    checkpoint_every = max(1, int(teacher_checkpoint_every))
    for step in range(int(steps)):
        pred = teacher(sx, dx, series_idx, cx)
        rank = pairwise_rank_loss(pred, left, right, sign, temperature=float(rank_temperature))
        huber = F.smooth_l1_loss(pred, targets)
        loss = rank + float(regression_weight) * huber
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0 or step == int(steps) - 1 or (step + 1) % max(1, int(steps) // 5) == 0:
            losses.append(
                {
                    "step": int(step + 1),
                    "teacher_total_loss": float(loss.detach().cpu().item()),
                    "teacher_rank_loss": float(rank.detach().cpu().item()),
                    "teacher_huber_loss": float(huber.detach().cpu().item()),
                    "teacher_pair_count": int(left.numel()),
                    "teacher_target": "u_comp_uniform",
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
                    device=device,
                )
                for name, split_rows in diagnostic_splits.items()
            }
            step_value = int(step + 1)
            checkpoint_history.append({"step": step_value, "diagnostics": diagnostics})
            checkpoint_states[step_value] = copy.deepcopy(teacher.state_dict())
    checkpoint_selection, selected_state = _selected_gipo_teacher_checkpoint(
        checkpoint_history,
        checkpoint_states,
        required_split_names=tuple(diagnostic_splits.keys()) if diagnostic_splits else (),
    )
    if selected_state is not None:
        teacher.load_state_dict(selected_state)
    return {
        "teacher_objective": "pairwise_rank_plus_huber_regression",
        "teacher_target": "u_comp_uniform",
        "teacher_density_feature": "train_normalized_log_density",
        "setting_feature_mode": feature_mode,
        "teacher_pair_count": int(left.numel()),
        "rank_temperature": float(rank_temperature),
        "regression_weight": float(regression_weight),
        "pair_margin": float(pair_margin),
        "losses": losses,
        "teacher_checkpoint_selection": checkpoint_selection,
        "fit_context_count": int(len(fit_context_ids)),
        "fit_series_count": int(len(fit_series_keys)),
    }


def _series_with_dynamic_unknown(base_series_idx: torch.Tensor, *, unknown_index: int, dropout: float) -> torch.Tensor:
    probability = min(1.0, max(0.0, float(dropout)))
    if probability <= 0.0 or base_series_idx.numel() == 0:
        return base_series_idx
    mask = torch.rand(base_series_idx.shape, device=base_series_idx.device) < probability
    out = base_series_idx.clone()
    out[mask] = int(unknown_index)
    return out


def build_teacher_weighted_density_targets(
    teacher: GIPOScheduleTeacherMLP,
    rows: Sequence[MetricRow],
    *,
    context_embeddings: Mapping[str, Sequence[float]],
    series_index_map: Mapping[str, int],
    schedule_grids: Mapping[ScheduleGridKey, Sequence[float]] | None,
    reference_time_grid: Sequence[float],
    density_normalizer: DensityFeatureNormalizer,
    supervision_schedule_keys: Sequence[str] | None = None,
    temperature: float = DEFAULT_TEACHER_TARGET_TEMPERATURE,
    temperature_mode: str = TEACHER_TEMPERATURE_MODE_FIXED,
    target_ess: float = DEFAULT_TEACHER_TARGET_ESS,
    min_temperature: float = DEFAULT_TEACHER_MIN_TEMPERATURE,
    max_temperature: float = DEFAULT_TEACHER_MAX_TEMPERATURE,
    student_target_mode: str = STUDENT_TARGET_MODE_SOFT_MIXTURE,
    teacher_hard_margin: float = DEFAULT_TEACHER_HARD_MARGIN,
    setting_feature_mode: str = SETTING_FEATURE_MODE_GIPO_V1,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    temperature_mode = validate_teacher_temperature_mode(temperature_mode)
    target_mode = validate_student_target_mode(student_target_mode)
    feature_mode = validate_setting_feature_mode(setting_feature_mode)
    fixed_temperature = _finite_positive(temperature, label="teacher_temperature")
    adaptive_target_ess = _finite_positive(target_ess, label="teacher_target_ess")
    adaptive_min_temperature = _finite_positive(min_temperature, label="teacher_min_temperature")
    adaptive_max_temperature = _finite_positive(max_temperature, label="teacher_max_temperature")
    hard_margin = float(teacher_hard_margin)
    if not math.isfinite(hard_margin) or hard_margin < 0.0:
        raise ValueError(f"teacher_hard_margin must be finite and nonnegative, got {teacher_hard_margin!r}.")
    if adaptive_min_temperature > adaptive_max_temperature:
        raise ValueError("teacher_min_temperature must be <= teacher_max_temperature.")
    observed_keys = {str(row["scheduler_key"]) for row in rows}
    supervision_keys = validate_gipo_support_schedule_keys(
        sorted(observed_keys) if supervision_schedule_keys is None else supervision_schedule_keys
    )
    supervision_set = set(supervision_keys)
    validate_gipo_support_schedule_keys(sorted(observed_keys))
    unsupported_observed = sorted(observed_keys - supervision_set)
    if unsupported_observed:
        raise ValueError(f"Rows contain schedules outside supervision_schedule_keys: {unsupported_observed}")
    grouped: Dict[ContextPairKey, List[MetricRow]] = defaultdict(list)
    for row in rows:
        grouped[context_pair_key(row, pair_on_seed=True)].append(row)
    if not grouped:
        raise ValueError("Student target construction requires at least one context group.")
    setting_rows: List[torch.Tensor] = []
    series_rows: List[int] = []
    context_rows: List[np.ndarray] = []
    target_masses: List[np.ndarray] = []
    entropy_values: List[float] = []
    ess_values: List[float] = []
    max_weight_values: List[float] = []
    chosen_temperature_values: List[float] = []
    top_margin_values: List[float] = []
    candidate_counts: List[int] = []
    hard_target_count = 0
    teacher.to(device)
    teacher.eval()
    for _, group in sorted(grouped.items(), key=lambda item: item[0]):
        counts: Dict[str, int] = {key: 0 for key in supervision_keys}
        for row in group:
            key = str(row["scheduler_key"])
            counts[key] = counts.get(key, 0) + 1
        bad_counts = {key: count for key, count in counts.items() if count != 1}
        if bad_counts:
            raise ValueError(f"Teacher-weighted density targets require exactly one row per supervision schedule; counts={bad_counts}.")
        first = group[0]
        context_id = context_id_from_row(first)
        if context_id not in context_embeddings:
            raise KeyError(f"Missing context embedding for {context_id}.")
        solver = str(first["solver_key"])
        target_nfe = int(first["target_nfe"])
        masses: List[Tuple[float, ...]] = []
        utilities: List[float] = []
        setting_row = setting_features(solver, target_nfe, mode=feature_mode)
        with torch.no_grad():
            for row in group:
                mass = density_mass_for_row(row, schedule_grids=schedule_grids, reference_time_grid=reference_time_grid)
                density_feature = density_normalizer.transform_one(mass, reference_time_grid=reference_time_grid)
                util = teacher(
                    setting_row[None, :].to(device=device),
                    torch.tensor(density_feature[None, :], dtype=torch.float32, device=device),
                    torch.tensor([remapped_series_index(row, series_index_map)], dtype=torch.long, device=device),
                    torch.tensor(np.asarray(context_embeddings[context_id], dtype=np.float32)[None, :], dtype=torch.float32, device=device),
                )[0]
                masses.append(mass)
                utilities.append(float(util.detach().cpu().item()))
        if temperature_mode == TEACHER_TEMPERATURE_MODE_ADAPTIVE_ESS:
            chosen_temperature = _teacher_temperature_for_target_ess(
                utilities,
                target_ess=adaptive_target_ess,
                min_temperature=adaptive_min_temperature,
                max_temperature=adaptive_max_temperature,
            )
        else:
            chosen_temperature = fixed_temperature
        utility_array = np.asarray(utilities, dtype=np.float64)
        if utility_array.size > 1:
            ordered = np.argsort(-utility_array)
            top_margin = float(utility_array[int(ordered[0])] - utility_array[int(ordered[1])])
        else:
            ordered = np.asarray([0], dtype=np.int64)
            top_margin = float("inf")
        if target_mode == STUDENT_TARGET_MODE_MARGIN_HARD_SOFT and top_margin >= hard_margin:
            weights = np.zeros_like(utility_array, dtype=np.float64)
            weights[int(ordered[0])] = 1.0
            hard_target_count += 1
        else:
            weights = _teacher_candidate_weights(utilities, temperature=chosen_temperature)
        mixture = np.zeros(len(reference_time_grid) - 1, dtype=np.float64)
        for weight, mass in zip(weights, masses):
            mixture += float(weight) * np.asarray(mass, dtype=np.float64)
        mixture = mixture / max(float(np.sum(mixture)), 1e-12)
        setting_rows.append(setting_row)
        series_rows.append(remapped_series_index(first, series_index_map))
        context_rows.append(np.asarray(context_embeddings[context_id], dtype=np.float32))
        target_masses.append(mixture.astype(np.float32))
        entropy_values.append(float(-np.sum(weights * np.log(np.maximum(weights, 1e-12)))))
        ess_values.append(_teacher_candidate_ess(weights))
        max_weight_values.append(float(np.max(weights)))
        chosen_temperature_values.append(float(chosen_temperature))
        top_margin_values.append(top_margin)
        candidate_counts.append(int(len(group)))
    entropy_stats = _summary_percentiles(entropy_values)
    ess_stats = _summary_percentiles(ess_values)
    max_weight_stats = _summary_percentiles(max_weight_values)
    chosen_temperature_stats = _summary_percentiles(chosen_temperature_values)
    top_margin_stats = _summary_percentiles(top_margin_values)
    summary = {
        "target_protocol": "teacher_weighted_density_mle",
        "student_target_mode": target_mode,
        "teacher_hard_margin": float(hard_margin),
        "hard_target_count": int(hard_target_count),
        "hard_target_fraction": float(hard_target_count / max(len(target_masses), 1)),
        "teacher_temperature_mode": temperature_mode,
        "teacher_temperature": float(fixed_temperature),
        "teacher_target_ess": float(adaptive_target_ess),
        "teacher_min_temperature": float(adaptive_min_temperature),
        "teacher_max_temperature": float(adaptive_max_temperature),
        "setting_feature_mode": feature_mode,
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
        "teacher_chosen_temperature_mean": chosen_temperature_stats["mean"],
        "teacher_chosen_temperature_p05": chosen_temperature_stats["p05"],
        "teacher_chosen_temperature_p50": chosen_temperature_stats["p50"],
        "teacher_chosen_temperature_p95": chosen_temperature_stats["p95"],
        "teacher_top_margin_mean": top_margin_stats["mean"],
        "teacher_top_margin_p05": top_margin_stats["p05"],
        "teacher_top_margin_p50": top_margin_stats["p50"],
        "teacher_top_margin_p95": top_margin_stats["p95"],
        "mean_candidate_count": float(np.mean(np.asarray(candidate_counts, dtype=np.float64))) if candidate_counts else 0.0,
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


def build_teacher_weighted_density_prediction_rows(
    teacher: GIPOScheduleTeacherMLP,
    rows: Sequence[MetricRow],
    *,
    context_embeddings: Mapping[str, Sequence[float]],
    series_index_map: Mapping[str, int],
    schedule_grids: Mapping[ScheduleGridKey, Sequence[float]] | None,
    reference_time_grid: Sequence[float],
    density_normalizer: DensityFeatureNormalizer,
    supervision_schedule_keys: Sequence[str] | None = None,
    temperature: float = DEFAULT_TEACHER_TARGET_TEMPERATURE,
    temperature_mode: str = TEACHER_TEMPERATURE_MODE_FIXED,
    target_ess: float = DEFAULT_TEACHER_TARGET_ESS,
    min_temperature: float = DEFAULT_TEACHER_MIN_TEMPERATURE,
    max_temperature: float = DEFAULT_TEACHER_MAX_TEMPERATURE,
    student_target_mode: str = STUDENT_TARGET_MODE_SOFT_MIXTURE,
    teacher_hard_margin: float = DEFAULT_TEACHER_HARD_MARGIN,
    setting_feature_mode: str = SETTING_FEATURE_MODE_GIPO_V1,
    device: torch.device | str = "cpu",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    temperature_mode = validate_teacher_temperature_mode(temperature_mode)
    target_mode = validate_student_target_mode(student_target_mode)
    feature_mode = validate_setting_feature_mode(setting_feature_mode)
    fixed_temperature = _finite_positive(temperature, label="teacher_temperature")
    adaptive_target_ess = _finite_positive(target_ess, label="teacher_target_ess")
    adaptive_min_temperature = _finite_positive(min_temperature, label="teacher_min_temperature")
    adaptive_max_temperature = _finite_positive(max_temperature, label="teacher_max_temperature")
    hard_margin = float(teacher_hard_margin)
    if not math.isfinite(hard_margin) or hard_margin < 0.0:
        raise ValueError(f"teacher_hard_margin must be finite and nonnegative, got {teacher_hard_margin!r}.")
    observed_keys = {str(row["scheduler_key"]) for row in rows}
    supervision_keys = validate_gipo_support_schedule_keys(
        sorted(observed_keys) if supervision_schedule_keys is None else supervision_schedule_keys
    )
    supervision_set = set(supervision_keys)
    unsupported_observed = sorted(observed_keys - supervision_set)
    if unsupported_observed:
        raise ValueError(f"Rows contain schedules outside supervision_schedule_keys: {unsupported_observed}")

    grouped: Dict[ContextPairKey, List[MetricRow]] = defaultdict(list)
    for row in rows:
        grouped[context_pair_key(row, pair_on_seed=True)].append(row)
    if not grouped:
        raise ValueError("Teacher oracle prediction requires at least one context group.")

    records: List[Dict[str, Any]] = []
    entropy_values: List[float] = []
    ess_values: List[float] = []
    max_weight_values: List[float] = []
    chosen_temperature_values: List[float] = []
    top_margin_values: List[float] = []
    candidate_counts: List[int] = []
    hard_target_count = 0
    teacher.to(device)
    teacher.eval()
    for _, group in sorted(grouped.items(), key=lambda item: item[0]):
        counts: Dict[str, int] = {key: 0 for key in supervision_keys}
        for row in group:
            key = str(row["scheduler_key"])
            counts[key] = counts.get(key, 0) + 1
        bad_counts = {key: count for key, count in counts.items() if count != 1}
        if bad_counts:
            raise ValueError(f"Teacher oracle prediction requires exactly one row per supervision schedule; counts={bad_counts}.")
        first = group[0]
        context_id = context_id_from_row(first)
        if context_id not in context_embeddings:
            raise KeyError(f"Missing context embedding for {context_id}.")
        solver = str(first["solver_key"])
        target_nfe = int(first["target_nfe"])
        setting_row = setting_features(solver, target_nfe, mode=feature_mode)
        masses: List[Tuple[float, ...]] = []
        utilities: List[float] = []
        schedule_keys: List[str] = []
        with torch.no_grad():
            for row in group:
                mass = density_mass_for_row(row, schedule_grids=schedule_grids, reference_time_grid=reference_time_grid)
                density_feature = density_normalizer.transform_one(mass, reference_time_grid=reference_time_grid)
                util = teacher(
                    setting_row[None, :].to(device=device),
                    torch.tensor(density_feature[None, :], dtype=torch.float32, device=device),
                    torch.tensor([remapped_series_index(row, series_index_map)], dtype=torch.long, device=device),
                    torch.tensor(np.asarray(context_embeddings[context_id], dtype=np.float32)[None, :], dtype=torch.float32, device=device),
                )[0]
                masses.append(mass)
                utilities.append(float(util.detach().cpu().item()))
                schedule_keys.append(str(row["scheduler_key"]))
        if temperature_mode == TEACHER_TEMPERATURE_MODE_ADAPTIVE_ESS:
            chosen_temperature = _teacher_temperature_for_target_ess(
                utilities,
                target_ess=adaptive_target_ess,
                min_temperature=adaptive_min_temperature,
                max_temperature=adaptive_max_temperature,
            )
        else:
            chosen_temperature = fixed_temperature
        utility_array = np.asarray(utilities, dtype=np.float64)
        if utility_array.size > 1:
            ordered = np.argsort(-utility_array)
            top_margin = float(utility_array[int(ordered[0])] - utility_array[int(ordered[1])])
        else:
            ordered = np.asarray([0], dtype=np.int64)
            top_margin = float("inf")
        hard_target = bool(target_mode == STUDENT_TARGET_MODE_MARGIN_HARD_SOFT and top_margin >= hard_margin)
        if hard_target:
            weights = np.zeros_like(utility_array, dtype=np.float64)
            weights[int(ordered[0])] = 1.0
            hard_target_count += 1
        else:
            weights = _teacher_candidate_weights(utilities, temperature=chosen_temperature)
        mixture = np.zeros(len(reference_time_grid) - 1, dtype=np.float64)
        for weight, mass in zip(weights, masses):
            mixture += float(weight) * np.asarray(mass, dtype=np.float64)
        mixture = mixture / max(float(np.sum(mixture)), 1e-12)
        macro_steps = solver_macro_steps(solver, target_nfe)
        grid = density_mass_to_time_grid(mixture, macro_steps=macro_steps, reference_time_grid=reference_time_grid)
        entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-12))))
        ess = _teacher_candidate_ess(weights)
        max_weight = float(np.max(weights))
        entropy_values.append(entropy)
        ess_values.append(ess)
        max_weight_values.append(max_weight)
        chosen_temperature_values.append(float(chosen_temperature))
        top_margin_values.append(top_margin)
        candidate_counts.append(int(len(group)))
        top_idx = int(ordered[0])
        copied = dict(first)
        copied.update(
            {
                "context_id": context_id,
                "solver_key": solver,
                "target_nfe": int(target_nfe),
                "macro_steps": int(macro_steps),
                "time_grid": list(grid),
                "schedule_grid_hash": schedule_grid_hash(grid),
                "density_mass": [float(x) for x in mixture.tolist()],
                "density_mass_hash": density_mass_hash(mixture, reference_time_grid=reference_time_grid),
                "density_protocol": DENSITY_PROTOCOL,
                "reference_grid_hash": reference_grid_hash(reference_time_grid),
                "setting_feature_mode": feature_mode,
                "teacher_temperature_mode": temperature_mode,
                "teacher_temperature": float(fixed_temperature),
                "teacher_target_ess": float(adaptive_target_ess),
                "teacher_chosen_temperature": float(chosen_temperature),
                "student_target_mode": target_mode,
                "teacher_hard_margin": float(hard_margin),
                "teacher_hard_target": hard_target,
                "teacher_top_margin": top_margin,
                "teacher_top_schedule_key": schedule_keys[top_idx],
                "teacher_candidate_entropy": entropy,
                "teacher_candidate_ess": ess,
                "teacher_candidate_max_weight": max_weight,
                "teacher_candidate_count": int(len(group)),
                "teacher_utilities_json": json.dumps(dict(zip(schedule_keys, utilities)), sort_keys=True, separators=(",", ":")),
                "teacher_weights_json": json.dumps({key: float(weight) for key, weight in zip(schedule_keys, weights)}, sort_keys=True, separators=(",", ":")),
            }
        )
        records.append(copied)

    entropy_stats = _summary_percentiles(entropy_values)
    ess_stats = _summary_percentiles(ess_values)
    max_weight_stats = _summary_percentiles(max_weight_values)
    chosen_temperature_stats = _summary_percentiles(chosen_temperature_values)
    top_margin_stats = _summary_percentiles(top_margin_values)
    summary = {
        "target_protocol": "teacher_weighted_density_mle",
        "student_target_mode": target_mode,
        "teacher_hard_margin": float(hard_margin),
        "hard_target_count": int(hard_target_count),
        "hard_target_fraction": float(hard_target_count / max(len(records), 1)),
        "teacher_temperature_mode": temperature_mode,
        "teacher_temperature": float(fixed_temperature),
        "teacher_target_ess": float(adaptive_target_ess),
        "teacher_min_temperature": float(adaptive_min_temperature),
        "teacher_max_temperature": float(adaptive_max_temperature),
        "setting_feature_mode": feature_mode,
        "context_setting_count": int(len(records)),
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
        "teacher_chosen_temperature_mean": chosen_temperature_stats["mean"],
        "teacher_chosen_temperature_p05": chosen_temperature_stats["p05"],
        "teacher_chosen_temperature_p50": chosen_temperature_stats["p50"],
        "teacher_chosen_temperature_p95": chosen_temperature_stats["p95"],
        "teacher_top_margin_mean": top_margin_stats["mean"],
        "teacher_top_margin_p05": top_margin_stats["p05"],
        "teacher_top_margin_p50": top_margin_stats["p50"],
        "teacher_top_margin_p95": top_margin_stats["p95"],
        "mean_candidate_count": float(np.mean(np.asarray(candidate_counts, dtype=np.float64))) if candidate_counts else 0.0,
        "supervision_schedule_keys": list(supervision_keys),
        "density_protocol": DENSITY_PROTOCOL,
    }
    return records, summary


def train_gipo_student(
    student: GIPODensityStudentMLP,
    teacher: GIPOScheduleTeacherMLP,
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
    teacher_temperature_mode: str = TEACHER_TEMPERATURE_MODE_FIXED,
    teacher_target_ess: float = DEFAULT_TEACHER_TARGET_ESS,
    teacher_min_temperature: float = DEFAULT_TEACHER_MIN_TEMPERATURE,
    teacher_max_temperature: float = DEFAULT_TEACHER_MAX_TEMPERATURE,
    student_target_mode: str = STUDENT_TARGET_MODE_SOFT_MIXTURE,
    teacher_hard_margin: float = DEFAULT_TEACHER_HARD_MARGIN,
    setting_feature_mode: str = SETTING_FEATURE_MODE_GIPO_V1,
    series_unknown_dropout: float = 0.10,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Student density training requires at least one fit row.")
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
        temperature_mode=str(teacher_temperature_mode),
        target_ess=float(teacher_target_ess),
        min_temperature=float(teacher_min_temperature),
        max_temperature=float(teacher_max_temperature),
        student_target_mode=str(student_target_mode),
        teacher_hard_margin=float(teacher_hard_margin),
        setting_feature_mode=str(setting_feature_mode),
        device=device,
    )
    opt = torch.optim.AdamW(student.parameters(), lr=float(lr), weight_decay=1e-4)
    losses: List[Dict[str, Any]] = []
    for step in range(int(steps)):
        series_idx = _series_with_dynamic_unknown(base_series_idx, unknown_index=student.unknown_series_index, dropout=float(series_unknown_dropout))
        logits = student.logits(sx, series_idx, cx)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(target_mass * log_probs).sum(dim=-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0 or step == int(steps) - 1 or (step + 1) % max(1, int(steps) // 5) == 0:
            with torch.no_grad():
                entropy = float((-(torch.softmax(logits, dim=-1) * log_probs).sum(dim=-1).mean()).detach().cpu().item())
            losses.append(
                {
                    "step": int(step + 1),
                    "student_kl_ce_loss": float(loss.detach().cpu().item()),
                    "student_entropy": entropy,
                }
            )
    return {
        "student_policy_type": "continuous_density",
        "student_objective": "teacher_weighted_density_mle_kl"
        if validate_student_target_mode(student_target_mode) == STUDENT_TARGET_MODE_SOFT_MIXTURE
        else "teacher_weighted_density_margin_hard_soft_kl",
        "density_protocol": DENSITY_PROTOCOL,
        "student_target_summary": target_summary,
        "series_unknown_dropout": float(series_unknown_dropout),
        "series_unknown_dropout_mode": "dynamic_per_step",
        "losses": losses,
    }


def predict_gipo_density(
    student: GIPODensityStudentMLP,
    *,
    row: MetricRow,
    context_embedding: Sequence[float],
    series_index_map: Mapping[str, int],
    reference_time_grid: Sequence[float],
    setting_feature_mode: str = SETTING_FEATURE_MODE_GIPO_V1,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    solver = str(row["solver_key"])
    target_nfe = int(row["target_nfe"])
    feature_mode = validate_setting_feature_mode(setting_feature_mode)
    macro_steps = solver_macro_steps(solver, target_nfe)
    student.to(device)
    student.eval()
    with torch.no_grad():
        mass_t = student.density_mass(
            setting_features(solver, target_nfe, mode=feature_mode)[None, :].to(device=device),
            torch.tensor([remapped_series_index(row, series_index_map)], dtype=torch.long, device=device),
            torch.tensor(np.asarray(context_embedding, dtype=np.float32)[None, :], dtype=torch.float32, device=device),
        )[0]
    mass = tuple(float(x) for x in mass_t.detach().cpu().numpy().astype(np.float64).tolist())
    grid = density_mass_to_time_grid(mass, macro_steps=macro_steps, reference_time_grid=reference_time_grid)
    return {
        "solver_key": solver,
        "target_nfe": int(target_nfe),
        "macro_steps": int(macro_steps),
        "time_grid": list(grid),
        "schedule_grid_hash": schedule_grid_hash(grid),
        "density_mass": [float(x) for x in mass],
        "density_mass_hash": density_mass_hash(mass, reference_time_grid=reference_time_grid),
        "setting_feature_mode": feature_mode,
        **density_metadata(reference_time_grid),
    }


def read_metric_rows_csv(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


__all__ = [
    "GIPO_PROTOCOL",
    "GIPO_SUPPORT_SCHEDULE_KEYS",
    "DEFAULT_CONTEXT_CALIBRATION_TOTAL",
    "DEFAULT_DENSITY_BIN_COUNT",
    "DEFAULT_SUPPORT_SCHEDULE_KEYS",
    "DEFAULT_SUPERVISION_SCHEDULE_KEYS",
    "EXPERIMENTAL_SUPERVISION_SCHEDULE_KEYS",
    "DEFAULT_TEACHER_HARD_MARGIN",
    "DEFAULT_TEACHER_MAX_TEMPERATURE",
    "DEFAULT_TEACHER_MIN_TEMPERATURE",
    "DEFAULT_TEACHER_TARGET_ESS",
    "DEFAULT_TEACHER_TARGET_TEMPERATURE",
    "MAX_CONTEXT_CALIBRATION_TOTAL",
    "MIN_CONTEXT_CALIBRATION_TOTAL",
    "STUDENT_TARGET_MODE_MARGIN_HARD_SOFT",
    "STUDENT_TARGET_MODE_SOFT_MIXTURE",
    "TEACHER_TEMPERATURE_MODE_ADAPTIVE_ESS",
    "TEACHER_TEMPERATURE_MODE_FIXED",
    "GIPODensityStudentMLP",
    "GIPOScheduleTeacherMLP",
    "DensityFeatureNormalizer",
    "EmbeddingNormalizer",
    "attach_uniform_gipo_rewards",
    "build_series_index_map",
    "build_teacher_weighted_density_prediction_rows",
    "build_teacher_weighted_density_targets",
    "context_calibration_train_val_counts",
    "context_id_from_row",
    "context_pair_key",
    "gipo_teacher_diagnostics",
    "density_mass_for_row",
    "grid_for_schedule",
    "load_context_embedding_table",
    "pairwise_rank_loss",
    "predict_gipo_density",
    "read_metric_rows_csv",
    "recommended_context_calibration_count",
    "remapped_series_index",
    "sample_context_ids_stratified",
    "save_context_embedding_table",
    "schedule_grid_hash",
    "series_key_from_row",
    "split_rows_by_context_holdout",
    "split_rows_by_series_holdout",
    "train_gipo_student",
    "train_gipo_teacher",
    "validate_gipo_support_schedule_keys",
    "validate_student_target_mode",
    "validate_teacher_temperature_mode",
    "validate_reference_grid",
]
