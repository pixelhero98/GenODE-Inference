from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from genode.conditional_opd.models import (
    grid_to_intervals,
    setting_features,
    solver_macro_steps,
    validate_time_grid,
)
from genode.conditional_opd.objectives import DEFAULT_REWARD_EPS, UNIFORM_SCHEDULE_KEY
from genode.conditional_opd.ser_ptg_reference import SER_PTG_SCHEDULE_KEY
from genode.schedule_transfer.diffusion_flow_schedules import BASELINE_SCHEDULE_KEYS, build_schedule_grid

MetricRow = Mapping[str, Any]
ContextPairKey = Tuple[str, str, int, str, int | None]
ScheduleGridKey = Tuple[str, str, int]

CONTEXT_CONDITIONAL_PROTOCOL = "context_conditional_opd_v1"
DEFAULT_SUPPORT_SCHEDULE_KEYS: Tuple[str, ...] = tuple(BASELINE_SCHEDULE_KEYS) + (SER_PTG_SCHEDULE_KEY,)
CONTEXT_SUPPORT_SCHEDULE_KEYS: Tuple[str, ...] = DEFAULT_SUPPORT_SCHEDULE_KEYS
DEFAULT_CONTEXT_CALIBRATION_TOTAL = 120
DEFAULT_CONTEXT_CALIBRATION_VALIDATION_FRACTION = 0.20
MIN_CONTEXT_CALIBRATION_TOTAL = 72
MAX_CONTEXT_CALIBRATION_TOTAL = 144
DEFAULT_SUPPORT_CHOICE_MARGIN = 0.001
DEFAULT_TEACHER_SELECTION_MIN_PAIRWISE_ACCURACY = 0.65
DEFAULT_TEACHER_SELECTION_MIN_SPEARMAN = 0.0


def _finite_positive(value: Any, *, label: str) -> float:
    val = float(value)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"{label} must be finite and positive, got {value!r}.")
    return val


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
    """Stable context key based only on historical-window identity."""
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


def _optional_int(value: Any) -> int | None:
    if value is None or str(value) == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def context_pair_key(row: MetricRow, *, pair_on_seed: bool = True) -> ContextPairKey:
    seed = _optional_int(row.get("seed")) if pair_on_seed else None
    return (
        str(row.get("dataset", row.get("dataset_key", ""))),
        str(row["solver_key"]),
        int(row["target_nfe"]),
        context_id_from_row(row),
        seed,
    )


def attach_uniform_context_rewards(
    rows: Iterable[MetricRow],
    *,
    support_schedule_keys: Sequence[str] | None = None,
    uniform_schedule_key: str = UNIFORM_SCHEDULE_KEY,
    pair_on_seed: bool = True,
    eps: float = DEFAULT_REWARD_EPS,
) -> List[Dict[str, Any]]:
    """Attach uniform-anchored rewards without crossing context keys.

    Rows are paired inside exact `(dataset, solver, nfe, context_id, seed)` cells
    by default. Set `pair_on_seed=False` only after explicit seed aggregation.
    """
    materialized = [dict(row) for row in rows]
    support_keys = validate_context_support_schedule_keys(
        CONTEXT_SUPPORT_SCHEDULE_KEYS if support_schedule_keys is None else support_schedule_keys
    )
    support = {str(key) for key in support_keys}
    uniform_key = str(uniform_schedule_key)
    grouped: Dict[ContextPairKey, List[Dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        schedule_key = str(row["scheduler_key"])
        if schedule_key not in support:
            raise ValueError(f"Unsupported context-conditional schedule row {schedule_key!r}; expected fixed/SER support.")
        row["context_id"] = context_id_from_row(row)
        grouped[context_pair_key(row, pair_on_seed=pair_on_seed)].append(row)

    out: List[Dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: item[0]):
        support_counts = Counter(str(row["scheduler_key"]) for row in group)
        duplicate_or_missing = {schedule: int(support_counts.get(schedule, 0)) for schedule in sorted(support) if int(support_counts.get(schedule, 0)) != 1}
        if duplicate_or_missing:
            raise ValueError(
                "Uniform-anchored context rewards require exactly one row for every support schedule "
                f"in paired context {key}; counts={duplicate_or_missing}."
            )
        observed_group_keys = set(support_counts)
        missing_group_keys = sorted(support - observed_group_keys)
        if missing_group_keys:
            raise ValueError(f"Context reward group {key} is missing support schedules: {missing_group_keys}")
        uniform_rows = [row for row in group if str(row["scheduler_key"]) == uniform_key]
        if len(uniform_rows) != 1:
            raise ValueError(
                "Uniform-anchored context rewards require exactly one uniform row "
                f"for paired context {key}, got {len(uniform_rows)}."
            )
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
                    "context_reward_protocol": CONTEXT_CONDITIONAL_PROTOCOL,
                    "reward_anchor_schedule_key": uniform_key,
                    "uniform_crps": float(uniform_crps),
                    "uniform_mase": float(uniform_mase),
                    "u_crps_uniform": u_crps,
                    "u_mase_uniform": u_mase,
                    "u_comp_uniform": float(0.5 * (u_crps + u_mase)),
                }
            )
            out.append(copied)
    return out


def validate_context_support_schedule_keys(
    support_schedule_keys: Sequence[str],
    *,
    allowed_schedule_keys: Sequence[str] = CONTEXT_SUPPORT_SCHEDULE_KEYS,
) -> Tuple[str, ...]:
    keys = tuple(str(key) for key in support_schedule_keys)
    if not keys:
        raise ValueError("support_schedule_keys must not be empty.")
    allowed = {str(key) for key in allowed_schedule_keys}
    bo_like = sorted(key for key in keys if "bo" in key.lower() or "candidate" in key.lower())
    if bo_like:
        raise ValueError(f"Context-conditional support must not include BO/candidate schedules: {bo_like}")
    unsupported = sorted(set(keys) - allowed)
    if unsupported:
        raise ValueError(f"Context-conditional support is fixed/SER only; unsupported schedules: {unsupported}")
    return keys


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
    """Return the capped context count for per-context fixed/SER calibration.

    The default keeps solar-style context-cell evaluation around 30k rows for
    7 schedules, 12 cells, and 3 seeds, instead of expanding to the full pool.
    """
    available = int(available_contexts)
    if available <= 0:
        raise ValueError("available_contexts must be positive.")
    cap = min(int(max_total), max(int(min_total), 12 * int(cells)))
    if normalized_combined_reference is None:
        requested = int(default_total)
    else:
        requested = int(round(0.20 * float(normalized_combined_reference)))
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
    """Sample contexts with deterministic series/temporal stratification."""
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
        temporal_bin = int(math.floor(float(target_t) / 24.0))
        strata[(series, temporal_bin)].append(context_id)

    rng = np.random.default_rng(int(seed))
    selected: List[str] = []
    stratum_items = sorted(strata.items(), key=lambda item: item[0])
    raw_allocations: List[Tuple[float, int, int]] = []
    for idx, (_, ids) in enumerate(stratum_items):
        raw = float(target) * float(len(ids)) / float(len(by_context))
        base = min(len(ids), int(math.floor(raw)))
        raw_allocations.append((raw - base, idx, base))
    counts = [base for _, _, base in raw_allocations]
    remaining = int(target - sum(counts))
    for _, idx, _ in sorted(raw_allocations, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        ids = stratum_items[idx][1]
        if counts[idx] < len(ids):
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


def save_context_embedding_table(
    path: str | Path,
    embeddings: Mapping[str, Sequence[float]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    sorted_ids = sorted(str(key) for key in embeddings)
    max_id_len = max(1, max(len(context_id) for context_id in sorted_ids))
    context_ids = np.asarray(sorted_ids, dtype=f"<U{max_id_len}")
    if context_ids.size == 0:
        raise ValueError("Cannot save an empty context embedding table.")
    matrix = np.asarray([embeddings[str(context_id)] for context_id in context_ids.tolist()], dtype=np.float32)
    np.savez_compressed(resolved, context_ids=context_ids, embeddings=matrix)
    manifest = {
        "artifact": "context_embedding_table",
        "protocol": CONTEXT_CONDITIONAL_PROTOCOL,
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


class ContextScheduleTeacherMLP(nn.Module):
    """Teacher scorer for `(solver, nfe, support interval, series, context)`."""

    def __init__(
        self,
        *,
        setting_dim: int,
        max_macro_steps: int,
        context_dim: int,
        num_series: int,
        series_embedding_dim: int = 32,
        hidden_dim: int = 256,
        hidden_layers: int = 3,
    ):
        super().__init__()
        self.max_macro_steps = int(max_macro_steps)
        self.context_dim = int(context_dim)
        self.num_series = int(num_series)
        self.unknown_series_index = int(num_series)
        self.series_embedding = nn.Embedding(int(num_series) + 1, int(series_embedding_dim))
        input_dim = int(setting_dim) + int(max_macro_steps) + int(context_dim) + int(series_embedding_dim)
        self.net = _mlp(input_dim, int(hidden_dim), 1, int(hidden_layers))

    def forward(
        self,
        setting_feature_batch: torch.Tensor,
        interval_batch: torch.Tensor,
        series_index_batch: torch.Tensor,
        context_embedding_batch: torch.Tensor,
    ) -> torch.Tensor:
        features = self._features(setting_feature_batch, interval_batch, series_index_batch, context_embedding_batch)
        return self.net(features).squeeze(-1)

    def _features(
        self,
        setting_feature_batch: torch.Tensor,
        interval_batch: torch.Tensor,
        series_index_batch: torch.Tensor,
        context_embedding_batch: torch.Tensor,
    ) -> torch.Tensor:
        if setting_feature_batch.ndim != 2:
            raise ValueError("setting_feature_batch must be 2D.")
        if interval_batch.ndim != 2 or interval_batch.shape[-1] != self.max_macro_steps:
            raise ValueError("interval_batch must be 2D with max_macro_steps columns.")
        if context_embedding_batch.ndim != 2 or context_embedding_batch.shape[-1] != self.context_dim:
            raise ValueError("context_embedding_batch must be 2D with context_dim columns.")
        series = series_index_batch.to(device=setting_feature_batch.device, dtype=torch.long).reshape(-1)
        series = torch.where(
            (series >= 0) & (series < self.num_series),
            series,
            torch.full_like(series, self.unknown_series_index),
        )
        series_emb = self.series_embedding(series)
        context_embedding_batch = context_embedding_batch.to(device=setting_feature_batch.device, dtype=setting_feature_batch.dtype)
        interval_batch = interval_batch.to(device=setting_feature_batch.device, dtype=setting_feature_batch.dtype)
        return torch.cat([setting_feature_batch, interval_batch, series_emb, context_embedding_batch], dim=-1)


class ContextSupportStudentMLP(nn.Module):
    """Categorical support policy over measured fixed/SER schedule keys."""

    def __init__(
        self,
        *,
        setting_dim: int,
        context_dim: int,
        num_series: int,
        support_schedule_keys: Sequence[str],
        series_embedding_dim: int = 32,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
    ):
        super().__init__()
        keys = validate_context_support_schedule_keys(support_schedule_keys)
        self.support_schedule_keys = keys
        self.context_dim = int(context_dim)
        self.num_series = int(num_series)
        self.unknown_series_index = int(num_series)
        self.series_embedding = nn.Embedding(int(num_series) + 1, int(series_embedding_dim))
        input_dim = int(setting_dim) + int(context_dim) + int(series_embedding_dim)
        self.net = _mlp(input_dim, int(hidden_dim), len(keys), int(hidden_layers))

    def logits(
        self,
        setting_feature_batch: torch.Tensor,
        series_index_batch: torch.Tensor,
        context_embedding_batch: torch.Tensor,
    ) -> torch.Tensor:
        if setting_feature_batch.ndim != 2:
            raise ValueError("setting_feature_batch must be 2D.")
        if context_embedding_batch.ndim != 2 or context_embedding_batch.shape[-1] != self.context_dim:
            raise ValueError("context_embedding_batch must be 2D with context_dim columns.")
        series = series_index_batch.to(device=setting_feature_batch.device, dtype=torch.long).reshape(-1)
        series = torch.where(
            (series >= 0) & (series < self.num_series),
            series,
            torch.full_like(series, self.unknown_series_index),
        )
        series_emb = self.series_embedding(series)
        context_embedding_batch = context_embedding_batch.to(device=setting_feature_batch.device, dtype=setting_feature_batch.dtype)
        return self.net(torch.cat([setting_feature_batch, series_emb, context_embedding_batch], dim=-1))

    def probabilities(
        self,
        setting_feature_batch: torch.Tensor,
        series_index_batch: torch.Tensor,
        context_embedding_batch: torch.Tensor,
    ) -> torch.Tensor:
        return F.softmax(self.logits(setting_feature_batch, series_index_batch, context_embedding_batch), dim=-1)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, hidden_layers: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    dim = int(input_dim)
    for _ in range(int(hidden_layers)):
        layers.extend([nn.Linear(dim, int(hidden_dim)), nn.SiLU()])
        dim = int(hidden_dim)
    layers.append(nn.Linear(dim, int(output_dim)))
    return nn.Sequential(*layers)


def padded_intervals_from_grid(
    solver_key: str,
    target_nfe: int,
    grid: Sequence[float],
    *,
    max_macro_steps: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    macro_steps = solver_macro_steps(str(solver_key), int(target_nfe))
    checked = validate_time_grid(grid, macro_steps=macro_steps)
    intervals = torch.zeros(int(max_macro_steps), dtype=torch.float32, device=device)
    raw = torch.tensor(grid_to_intervals(checked), dtype=torch.float32, device=device)
    intervals[: raw.numel()] = raw
    return intervals


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
    if key in BASELINE_SCHEDULE_KEYS:
        grid = build_schedule_grid(key, macro_steps)
        if grid is None:
            raise ValueError(f"No fixed schedule grid for {key}.")
        return validate_time_grid(grid, macro_steps=macro_steps)
    raise KeyError(f"Missing schedule grid for {(key, str(solver_key), int(target_nfe))}.")


def _teacher_training_tensors(
    rows: Sequence[MetricRow],
    *,
    context_embeddings: Mapping[str, Sequence[float]],
    series_index_map: Mapping[str, int],
    schedule_grids: Mapping[ScheduleGridKey, Sequence[float]] | None,
    max_macro_steps: int,
    device: torch.device | str,
    series_unknown_probability: float = 0.0,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[ContextPairKey], List[str]]:
    setting_rows: List[torch.Tensor] = []
    interval_rows: List[torch.Tensor] = []
    series_rows: List[int] = []
    context_rows: List[np.ndarray] = []
    targets: List[float] = []
    pair_keys: List[ContextPairKey] = []
    schedule_keys: List[str] = []
    rng = np.random.default_rng(int(seed))
    unknown_index = int(len(series_index_map))
    unknown_probability = min(1.0, max(0.0, float(series_unknown_probability)))
    for row in rows:
        context_id = context_id_from_row(row)
        if context_id not in context_embeddings:
            raise KeyError(f"Missing context embedding for {context_id}.")
        solver = str(row["solver_key"])
        target_nfe = int(row["target_nfe"])
        schedule_key = str(row["scheduler_key"])
        grid = grid_for_schedule(schedule_key, solver, target_nfe, schedule_grids=schedule_grids)
        setting_rows.append(setting_features(solver, target_nfe))
        interval_rows.append(
            padded_intervals_from_grid(
                solver,
                target_nfe,
                grid,
                max_macro_steps=int(max_macro_steps),
            )
        )
        series_idx = remapped_series_index(row, series_index_map)
        if unknown_probability > 0.0 and rng.random() < unknown_probability:
            series_idx = unknown_index
        series_rows.append(series_idx)
        context_rows.append(np.asarray(context_embeddings[context_id], dtype=np.float32))
        targets.append(float(row["u_comp_uniform"]))
        pair_keys.append(context_pair_key(row, pair_on_seed=True))
        schedule_keys.append(schedule_key)
    return (
        torch.stack(setting_rows, dim=0).to(device=device),
        torch.stack(interval_rows, dim=0).to(device=device),
        torch.tensor(series_rows, dtype=torch.long, device=device),
        torch.tensor(np.stack(context_rows, axis=0), dtype=torch.float32, device=device),
        torch.tensor(targets, dtype=torch.float32, device=device),
        pair_keys,
        schedule_keys,
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
    temp = max(float(temperature), 1e-6)
    logits = sign * (pred[left] - pred[right]) / temp
    return F.softplus(-logits).mean()


def _rank_values_desc(values: Mapping[str, float]) -> Dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (-float(item[1]), str(item[0])))
    ranks: Dict[str, float] = {}
    pos = 0
    while pos < len(ordered):
        end = pos + 1
        while end < len(ordered) and float(ordered[end][1]) == float(ordered[pos][1]):
            end += 1
        avg_rank = float(pos + end - 1) / 2.0
        for idx in range(pos, end):
            ranks[str(ordered[idx][0])] = avg_rank
        pos = end
    return ranks


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


def _support_choice_metrics(
    pred_values: Sequence[float],
    target_values: Sequence[float],
    pair_keys: Sequence[ContextPairKey],
    schedule_keys: Sequence[str],
) -> Dict[str, Any]:
    by_key_schedule: Dict[Tuple[str, str, int, str], Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for idx, key in enumerate(pair_keys):
        seed_mean_key = (str(key[0]), str(key[1]), int(key[2]), str(key[3]))
        by_key_schedule[seed_mean_key][str(schedule_keys[idx])].append(int(idx))
    correct = 0
    top2_hits = 0
    total = 0
    spearman_values: List[float] = []
    for schedule_to_indices in by_key_schedule.values():
        if len(schedule_to_indices) < 2:
            continue
        pred_by_schedule = {
            schedule: float(np.mean(np.asarray([float(pred_values[idx]) for idx in indices], dtype=np.float64)))
            for schedule, indices in schedule_to_indices.items()
        }
        target_by_schedule = {
            schedule: float(np.mean(np.asarray([float(target_values[idx]) for idx in indices], dtype=np.float64)))
            for schedule, indices in schedule_to_indices.items()
        }
        actual_order = sorted(target_by_schedule, key=lambda schedule: (-float(target_by_schedule[schedule]), str(schedule)))
        pred_order = sorted(pred_by_schedule, key=lambda schedule: (-float(pred_by_schedule[schedule]), str(schedule)))
        actual_top1 = str(actual_order[0])
        pred_top1 = str(pred_order[0])
        pred_top2 = {str(schedule) for schedule in pred_order[:2]}
        correct += int(actual_top1 == pred_top1)
        top2_hits += int(actual_top1 in pred_top2)
        spearman_values.append(_spearman_for_scores(pred_by_schedule, target_by_schedule))
        total += 1
    if total <= 0:
        return {
            "support_top1_accuracy": None,
            "support_top2_recall": None,
            "spearman_rank_correlation": None,
            "support_choice_group_count": 0,
        }
    return {
        "support_top1_accuracy": float(correct / total),
        "support_top2_recall": float(top2_hits / total),
        "spearman_rank_correlation": float(np.mean(np.asarray(spearman_values, dtype=np.float64))) if spearman_values else None,
        "support_choice_group_count": int(total),
    }


def context_teacher_diagnostics(
    teacher: ContextScheduleTeacherMLP,
    rows: Sequence[MetricRow],
    *,
    context_embeddings: Mapping[str, Sequence[float]],
    series_index_map: Mapping[str, int],
    schedule_grids: Mapping[ScheduleGridKey, Sequence[float]] | None = None,
    max_macro_steps: int = 12,
    rank_temperature: float = 0.5,
    regression_weight: float = 0.25,
    pair_margin: float = 0.0,
    pair_on_seed: bool = True,
    split_name: str = "diagnostic",
    fit_context_ids: Iterable[str] = (),
    fit_series_keys: Iterable[str] = (),
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    contexts = {context_id_from_row(row) for row in rows}
    series_keys = {series_key_from_row(row) for row in rows}
    schedules = {str(row["scheduler_key"]) for row in rows}
    split_phases = sorted({str(row.get("split_phase", row.get("split", ""))) for row in rows})
    fit_context_set = {str(value) for value in fit_context_ids}
    fit_series_set = {str(value) for value in fit_series_keys}
    base: Dict[str, Any] = {
        "split_name": str(split_name),
        "row_count": int(len(rows)),
        "context_count": int(len(contexts)),
        "series_count": int(len(series_keys)),
        "schedule_count": int(len(schedules)),
        "schedule_keys": sorted(schedules),
        "split_phases": split_phases,
        "fit_context_overlap_count": int(len(contexts & fit_context_set)),
        "fit_series_overlap_count": int(len(series_keys & fit_series_set)),
        "uses_validation_labels": False,
    }
    if not rows:
        base.update(
            {
                "rank_loss": None,
                "huber_loss": None,
                "total_loss": None,
                "pairwise_accuracy": None,
                "pair_count": 0,
                "top1_schedule_agreement": None,
                "top1_group_count": 0,
                "support_top1_accuracy": None,
                "support_top2_recall": None,
                "spearman_rank_correlation": None,
                "support_choice_group_count": 0,
            }
        )
        return base

    was_training = bool(teacher.training)
    teacher.eval()
    with torch.no_grad():
        sx, ix, series_idx, cx, targets, pair_keys, schedule_keys = _teacher_training_tensors(
            rows,
            context_embeddings=context_embeddings,
            series_index_map=series_index_map,
            schedule_grids=schedule_grids,
            max_macro_steps=int(max_macro_steps),
            device=device,
        )
        if not pair_on_seed:
            pair_keys = [key[:4] + (None,) for key in pair_keys]
        left, right, sign = _pair_indices(targets, pair_keys, margin=float(pair_margin), device=sx.device)
        pred = teacher(sx, ix, series_idx, cx)
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
        support_metrics = _support_choice_metrics(pred_values, target_values, pair_keys, schedule_keys)
    if was_training:
        teacher.train()
    base.update(
        {
            "rank_loss": float(rank.detach().cpu().item()),
            "huber_loss": float(huber.detach().cpu().item()),
            "total_loss": float(total.detach().cpu().item()),
            "pairwise_accuracy": pairwise_accuracy,
            "pair_count": int(left.numel()),
            "top1_schedule_agreement": support_metrics["support_top1_accuracy"],
            "top1_group_count": int(support_metrics["support_choice_group_count"]),
            **support_metrics,
        }
    )
    return base


def _selected_context_teacher_checkpoint(
    checkpoint_history: Sequence[Dict[str, Any]],
    checkpoint_states: Mapping[int, Mapping[str, torch.Tensor]],
    *,
    required_split_names: Sequence[str] = (),
    min_pairwise_accuracy: float = DEFAULT_TEACHER_SELECTION_MIN_PAIRWISE_ACCURACY,
    min_spearman: float = DEFAULT_TEACHER_SELECTION_MIN_SPEARMAN,
) -> Tuple[Dict[str, Any], Mapping[str, torch.Tensor] | None]:
    if not checkpoint_history:
        return (
            {
                "selection_protocol": "final_checkpoint_no_diagnostics",
                "selection_split": "none",
                "selected_step": None,
                "history": [],
                "uses_validation_labels": False,
            },
            None,
        )
    split_names = sorted(
        {
            str(name)
            for entry in checkpoint_history
            for name, diag in dict(entry.get("diagnostics", {})).items()
            if diag.get("support_top1_accuracy") is not None
        }
    )
    missing_required = [str(name) for name in required_split_names if str(name) not in split_names]
    if missing_required:
        raise ValueError(f"Teacher checkpoint selection requires usable diagnostics for all splits; missing {missing_required}.")
    scored_history: List[Dict[str, Any]] = []
    best_entry: Dict[str, Any] | None = None
    best_key: Tuple[float, float, float, float, float, int] | None = None
    constraints = {
        "min_pairwise_accuracy": float(min_pairwise_accuracy),
        "min_spearman": float(min_spearman),
    }
    for entry in checkpoint_history:
        copied = dict(entry)
        diagnostics = {str(k): dict(v) for k, v in dict(entry.get("diagnostics", {})).items()}
        pairwise_accuracies: List[float] = []
        spearman_values: List[float] = []
        huber_values: List[float] = []
        top1_values: List[float] = []
        top2_values: List[float] = []
        constraint_failures: List[str] = []
        for split in split_names:
            diag = diagnostics.get(split, {})
            top1 = diag.get("support_top1_accuracy")
            top2 = diag.get("support_top2_recall")
            pairwise = diag.get("pairwise_accuracy")
            spearman = diag.get("spearman_rank_correlation")
            huber = diag.get("huber_loss")
            if top1 is None or top2 is None:
                constraint_failures.append(f"{split}:missing_support_choice_metric")
            else:
                top1_values.append(float(top1))
                top2_values.append(float(top2))
            if pairwise is None or float(pairwise) < float(min_pairwise_accuracy):
                constraint_failures.append(f"{split}:pairwise_accuracy_below_{float(min_pairwise_accuracy):.4f}")
            else:
                pairwise_accuracies.append(float(pairwise))
            if spearman is None or float(spearman) <= float(min_spearman):
                constraint_failures.append(f"{split}:spearman_not_above_{float(min_spearman):.4f}")
            else:
                spearman_values.append(float(spearman))
            if huber is not None:
                huber_values.append(float(huber))
        worst_acc = float(min(pairwise_accuracies)) if pairwise_accuracies else 0.0
        support_choice_terms = top1_values + top2_values
        support_choice_score = (
            float(np.mean(np.asarray(support_choice_terms, dtype=np.float64)))
            if len(support_choice_terms) == 2 * len(split_names)
            else None
        )
        worst_top1 = float(min(top1_values)) if top1_values else 0.0
        mean_pairwise = float(np.mean(np.asarray(pairwise_accuracies, dtype=np.float64))) if pairwise_accuracies else 0.0
        mean_spearman = float(np.mean(np.asarray(spearman_values, dtype=np.float64))) if spearman_values else 0.0
        mean_huber = float(np.mean(np.asarray(huber_values, dtype=np.float64))) if huber_values else float("inf")
        thresholds_passed = support_choice_score is not None and not constraint_failures
        copied["diagnostics"] = diagnostics
        copied["worst_split_pairwise_accuracy"] = worst_acc
        copied["support_choice_score"] = support_choice_score
        copied["worst_split_support_top1_accuracy"] = worst_top1
        copied["mean_pairwise_accuracy"] = mean_pairwise
        copied["mean_spearman_rank_correlation"] = mean_spearman
        copied["mean_huber_loss"] = mean_huber
        copied["selection_constraints_passed"] = bool(thresholds_passed)
        copied["selection_constraint_failures"] = constraint_failures
        scored_history.append(copied)
        if not thresholds_passed:
            continue
        key = (
            -float(support_choice_score),
            -worst_top1,
            -mean_pairwise,
            -mean_spearman,
            mean_huber,
            int(copied["step"]),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_entry = copied
    if best_entry is None:
        raise ValueError(
            "Teacher checkpoint selection found no checkpoint satisfying support-choice constraints; "
            f"constraints={constraints}, scored_steps={[entry.get('step') for entry in scored_history]}."
        )
    selected_step = int(best_entry["step"])
    selected_state = checkpoint_states.get(selected_step)
    return (
        {
            "selection_protocol": "context_series_support_choice_teacher_checkpoint",
            "selection_split": "context_and_series_teacher_holdouts",
            "selected_step": int(selected_step),
            "selection_metric": "context_series_support_top1_top2",
            "selection_constraints": constraints,
            "selected_support_choice_score": best_entry.get("support_choice_score"),
            "tie_breaker": "higher_worst_split_top1_then_pairwise_then_spearman_then_lower_huber_then_earlier_step",
            "uses_validation_labels": False,
            "history": scored_history,
        },
        selected_state,
    )


def train_context_teacher(
    teacher: ContextScheduleTeacherMLP,
    rows: Sequence[MetricRow],
    *,
    context_embeddings: Mapping[str, Sequence[float]],
    series_index_map: Mapping[str, int],
    schedule_grids: Mapping[ScheduleGridKey, Sequence[float]] | None = None,
    max_macro_steps: int = 12,
    steps: int = 500,
    lr: float = 1e-3,
    rank_temperature: float = 0.5,
    regression_weight: float = 0.25,
    pair_margin: float = 0.0,
    pair_on_seed: bool = True,
    require_rank_pairs: bool = True,
    diagnostic_splits: Mapping[str, Sequence[MetricRow]] | None = None,
    teacher_checkpoint_every: int = 100,
    teacher_selection_min_pairwise_accuracy: float = DEFAULT_TEACHER_SELECTION_MIN_PAIRWISE_ACCURACY,
    teacher_selection_min_spearman: float = DEFAULT_TEACHER_SELECTION_MIN_SPEARMAN,
    series_unknown_probability: float = 0.0,
    seed: int = 0,
    allowed_schedule_keys: Sequence[str] = CONTEXT_SUPPORT_SCHEDULE_KEYS,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Teacher training requires at least one context reward row.")
    validate_context_support_schedule_keys(
        sorted({str(row["scheduler_key"]) for row in rows}),
        allowed_schedule_keys=allowed_schedule_keys,
    )
    teacher.to(device)
    sx, ix, series_idx, cx, targets, pair_keys, _ = _teacher_training_tensors(
        rows,
        context_embeddings=context_embeddings,
        series_index_map=series_index_map,
        schedule_grids=schedule_grids,
        max_macro_steps=int(max_macro_steps),
        device=device,
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
        pred = teacher(sx, ix, series_idx, cx)
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
            step_value = int(step + 1)
            diagnostics = {
                str(name): context_teacher_diagnostics(
                    teacher,
                    split_rows,
                    context_embeddings=context_embeddings,
                    series_index_map=series_index_map,
                    schedule_grids=schedule_grids,
                    max_macro_steps=int(max_macro_steps),
                    rank_temperature=float(rank_temperature),
                    regression_weight=float(regression_weight),
                    pair_margin=float(pair_margin),
                    pair_on_seed=bool(pair_on_seed),
                    split_name=str(name),
                    fit_context_ids=fit_context_ids,
                    fit_series_keys=fit_series_keys,
                    device=device,
                )
                for name, split_rows in diagnostic_splits.items()
            }
            checkpoint_history.append({"step": step_value, "diagnostics": diagnostics})
            checkpoint_states[step_value] = copy.deepcopy(teacher.state_dict())
    checkpoint_selection, selected_state = _selected_context_teacher_checkpoint(
        checkpoint_history,
        checkpoint_states,
        required_split_names=tuple(diagnostic_splits or ()),
        min_pairwise_accuracy=float(teacher_selection_min_pairwise_accuracy),
        min_spearman=float(teacher_selection_min_spearman),
    )
    if selected_state is not None:
        teacher.load_state_dict(selected_state)
    return {
        "losses": losses,
        "teacher_pair_count": int(left.numel()),
        "teacher_row_count": int(len(rows)),
        "context_count": int(len({context_id_from_row(row) for row in rows})),
        "reward_protocol": CONTEXT_CONDITIONAL_PROTOCOL,
        "series_unknown_probability": float(series_unknown_probability),
        "checkpoint_selection": checkpoint_selection,
        "_selected_state_dict": selected_state,
    }


def support_student_ce_loss(student_logits: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
    if student_logits.ndim != 2 or target_distribution.ndim != 2:
        raise ValueError("student_logits and target_distribution must be 2D tensors.")
    if student_logits.shape != target_distribution.shape:
        raise ValueError("student_logits and target_distribution shapes must match.")
    log_probs = F.log_softmax(student_logits, dim=-1)
    return -(target_distribution.to(device=student_logits.device, dtype=student_logits.dtype) * log_probs).sum(dim=-1).mean()


def _context_setting_key(row: MetricRow) -> Tuple[str, str, int, str]:
    return (
        str(row.get("dataset", row.get("dataset_key", ""))),
        str(row["solver_key"]),
        int(row["target_nfe"]),
        context_id_from_row(row),
    )


def _support_groups_by_context_setting(
    rows: Sequence[MetricRow],
    support_schedule_keys: Sequence[str],
) -> List[Tuple[Tuple[str, str, int, str], Dict[str, List[Dict[str, Any]]], Dict[str, Any]]]:
    support = tuple(str(key) for key in support_schedule_keys)
    grouped: Dict[Tuple[str, str, int, str], Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    refs: Dict[Tuple[str, str, int, str], Dict[str, Any]] = {}
    for row in rows:
        schedule = str(row["scheduler_key"])
        if schedule not in set(support):
            continue
        key = _context_setting_key(row)
        copied = dict(row)
        copied["context_id"] = key[3]
        grouped[key][schedule].append(copied)
        refs.setdefault(key, copied)
    out: List[Tuple[Tuple[str, str, int, str], Dict[str, List[Dict[str, Any]]], Dict[str, Any]]] = []
    for key in sorted(grouped):
        schedule_rows = {schedule: list(grouped[key].get(schedule, [])) for schedule in support}
        missing = [schedule for schedule, items in schedule_rows.items() if not items]
        if missing:
            raise ValueError(f"Student support target group {key} is missing support schedules: {missing}")
        out.append((key, schedule_rows, refs[key]))
    return out


def _mean_observed_utility(rows: Sequence[Mapping[str, Any]], *, schedule_key: str, context_key: Tuple[str, str, int, str]) -> float:
    values = [float(row["u_comp_uniform"]) for row in rows if row.get("u_comp_uniform") not in (None, "")]
    if not values:
        raise ValueError(f"Missing observed u_comp_uniform for schedule {schedule_key!r} in context setting {context_key}.")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _apply_series_unknown_dropout(
    series_idx: torch.Tensor,
    *,
    unknown_index: int,
    probability: float,
) -> torch.Tensor:
    p = min(1.0, max(0.0, float(probability)))
    if p <= 0.0 or series_idx.numel() == 0:
        return series_idx
    mask = torch.rand(series_idx.shape, device=series_idx.device) < p
    return torch.where(mask, torch.full_like(series_idx, int(unknown_index)), series_idx)


def build_teacher_guided_support_targets(
    teacher: ContextScheduleTeacherMLP,
    rows: Sequence[MetricRow],
    *,
    context_embeddings: Mapping[str, Sequence[float]],
    series_index_map: Mapping[str, int],
    support_schedule_keys: Sequence[str],
    schedule_grids: Mapping[ScheduleGridKey, Sequence[float]] | None = None,
    max_macro_steps: int = 12,
    support_choice_margin: float = DEFAULT_SUPPORT_CHOICE_MARGIN,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    support_keys = validate_context_support_schedule_keys(tuple(str(key) for key in support_schedule_keys))
    groups = _support_groups_by_context_setting(rows, support_keys)
    if not groups:
        raise ValueError("Student target construction requires at least one complete context/support group.")
    teacher.to(device)
    teacher.eval()
    setting_rows: List[torch.Tensor] = []
    series_rows: List[int] = []
    context_rows: List[np.ndarray] = []
    target_rows: List[np.ndarray] = []
    target_source_counts: Counter[str] = Counter()
    selected_target_counts: Counter[str] = Counter()
    teacher_top1_counts: Counter[str] = Counter()
    observed_top1_counts: Counter[str] = Counter()
    margin = float(support_choice_margin)
    with torch.no_grad():
        for key, schedule_rows, ref in groups:
            _, solver, target_nfe, context_id = key
            if context_id not in context_embeddings:
                raise KeyError(f"Missing context embedding for {context_id}.")
            setting = setting_features(solver, target_nfe)
            context_np = np.asarray(context_embeddings[context_id], dtype=np.float32)
            context = torch.tensor(context_np[None, :], dtype=torch.float32, device=device)
            series_value = remapped_series_index(ref, series_index_map)
            series = torch.tensor([series_value], dtype=torch.long, device=device)
            observed_utilities = {
                schedule: _mean_observed_utility(schedule_rows[schedule], schedule_key=schedule, context_key=key)
                for schedule in support_keys
            }
            teacher_utilities: Dict[str, float] = {}
            for schedule in support_keys:
                grid = grid_for_schedule(schedule, solver, target_nfe, schedule_grids=schedule_grids)
                interval = padded_intervals_from_grid(
                    solver,
                    target_nfe,
                    grid,
                    max_macro_steps=int(max_macro_steps),
                    device=device,
                )[None, :]
                teacher_utilities[schedule] = float(teacher(setting[None, :].to(device), interval, series, context)[0].detach().cpu().item())
            teacher_order = sorted(support_keys, key=lambda schedule: (-float(teacher_utilities[schedule]), str(schedule)))
            observed_order = sorted(support_keys, key=lambda schedule: (-float(observed_utilities[schedule]), str(schedule)))
            teacher_top1 = str(teacher_order[0])
            observed_top1 = str(observed_order[0])
            teacher_top1_counts[teacher_top1] += 1
            observed_top1_counts[observed_top1] += 1
            teacher_accepted = float(observed_utilities[teacher_top1]) >= float(observed_utilities[observed_top1]) - margin
            source = "teacher_accepted" if teacher_accepted else "observed_fallback"
            source_order = teacher_order if teacher_accepted else observed_order
            source_values = teacher_utilities if teacher_accepted else observed_utilities
            top1 = str(source_order[0])
            target = np.zeros(len(support_keys), dtype=np.float32)
            if len(source_order) > 1 and float(source_values[top1]) - float(source_values[str(source_order[1])]) <= margin:
                top2 = str(source_order[1])
                target[support_keys.index(top1)] = 0.6
                target[support_keys.index(top2)] = 0.4
                target_source_counts[f"{source}_top2"] += 1
            else:
                target[support_keys.index(top1)] = 1.0
                target_source_counts[f"{source}_top1"] += 1
            selected_target_counts[top1] += 1
            setting_rows.append(setting)
            series_rows.append(int(series_value))
            context_rows.append(context_np)
            target_rows.append(target)
    return (
        torch.stack(setting_rows, dim=0).to(device=device),
        torch.tensor(series_rows, dtype=torch.long, device=device),
        torch.tensor(np.stack(context_rows, axis=0), dtype=torch.float32, device=device),
        torch.tensor(np.stack(target_rows, axis=0), dtype=torch.float32, device=device),
        {
            "target_protocol": "teacher_guided_top1_top2_support_ce",
            "support_choice_margin": float(margin),
            "context_setting_count": int(len(groups)),
            "support_schedule_keys": list(support_keys),
            "target_source_counts": {key: int(value) for key, value in sorted(target_source_counts.items())},
            "selected_target_counts": {key: int(value) for key, value in sorted(selected_target_counts.items())},
            "teacher_top1_counts": {key: int(value) for key, value in sorted(teacher_top1_counts.items())},
            "observed_top1_counts": {key: int(value) for key, value in sorted(observed_top1_counts.items())},
        },
    )


def train_context_support_student(
    student: ContextSupportStudentMLP,
    teacher: ContextScheduleTeacherMLP,
    rows: Sequence[MetricRow],
    *,
    context_embeddings: Mapping[str, Sequence[float]],
    series_index_map: Mapping[str, int],
    support_schedule_keys: Sequence[str],
    schedule_grids: Mapping[ScheduleGridKey, Sequence[float]] | None = None,
    max_macro_steps: int = 12,
    steps: int = 500,
    lr: float = 1e-3,
    support_choice_margin: float = DEFAULT_SUPPORT_CHOICE_MARGIN,
    series_unknown_probability: float = 0.0,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    support_keys = tuple(str(key) for key in support_schedule_keys)
    support_keys = validate_context_support_schedule_keys(support_keys)
    if tuple(student.support_schedule_keys) != support_keys:
        raise ValueError("Student support_schedule_keys do not match training support.")
    student.to(device)
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    sx, base_series_idx, cx, target_probs, target_summary = build_teacher_guided_support_targets(
        teacher,
        rows,
        context_embeddings=context_embeddings,
        series_index_map=series_index_map,
        support_schedule_keys=support_keys,
        schedule_grids=schedule_grids,
        max_macro_steps=int(max_macro_steps),
        support_choice_margin=float(support_choice_margin),
        device=device,
    )
    unknown_index = int(len(series_index_map))
    unknown_probability = min(1.0, max(0.0, float(series_unknown_probability)))
    opt = torch.optim.AdamW(student.parameters(), lr=float(lr), weight_decay=1e-4)
    losses: List[Dict[str, Any]] = []
    torch.manual_seed(int(seed))
    for step in range(int(steps)):
        train_series_idx = _apply_series_unknown_dropout(
            base_series_idx,
            unknown_index=unknown_index,
            probability=unknown_probability,
        )
        logits = student.logits(sx, train_series_idx, cx)
        loss = support_student_ce_loss(logits, target_probs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0 or step == int(steps) - 1 or (step + 1) % max(1, int(steps) // 5) == 0:
            with torch.no_grad():
                probs = F.softmax(student.logits(sx, base_series_idx, cx), dim=-1)
                entropy = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=-1).mean()
            losses.append(
                {
                    "step": int(step + 1),
                    "student_ce_loss": float(loss.detach().cpu().item()),
                    "student_mean_entropy": float(entropy.detach().cpu().item()),
                    "student_objective": "teacher_guided_top1_top2_categorical_ce",
                }
            )
    return {
        "losses": losses,
        "student_context_setting_count": int(target_summary["context_setting_count"]),
        "support_schedule_keys": list(support_keys),
        "student_policy_type": "categorical_support",
        "student_objective": "teacher_guided_top1_top2_categorical_ce",
        "student_target_summary": target_summary,
        "reward_protocol": CONTEXT_CONDITIONAL_PROTOCOL,
        "series_unknown_probability": float(series_unknown_probability),
        "series_unknown_dropout_mode": "dynamic_per_step",
    }


def build_calibration_holdout_non_regression_guard(
    student: ContextSupportStudentMLP,
    rows: Sequence[MetricRow],
    *,
    context_embeddings: Mapping[str, Sequence[float]],
    series_index_map: Mapping[str, int],
    support_schedule_keys: Sequence[str],
    margin: float = DEFAULT_SUPPORT_CHOICE_MARGIN,
    source_holdout_names: Sequence[str] = ("context_disjoint", "series_disjoint"),
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    support_keys = validate_context_support_schedule_keys(tuple(str(key) for key in support_schedule_keys))
    if tuple(student.support_schedule_keys) != support_keys:
        raise ValueError("Student support_schedule_keys do not match calibration guard support.")
    if not rows:
        raise ValueError("Calibration non-regression guard requires calibration holdout rows.")
    locked_rows = [row for row in rows if str(row.get("split_phase", row.get("split", ""))) == "locked_test"]
    if locked_rows:
        raise ValueError(f"Calibration non-regression guard refuses locked_test rows; found {len(locked_rows)}.")
    expected_holdouts = {str(name) for name in source_holdout_names}
    observed_holdouts = {str(row.get("calibration_holdout_name", "") or "") for row in rows}
    if "" in observed_holdouts or not observed_holdouts:
        raise ValueError("Calibration guard rows require explicit calibration_holdout_name provenance.")
    unknown_holdouts = sorted(observed_holdouts - expected_holdouts)
    if unknown_holdouts:
        raise ValueError(f"Calibration guard rows contain unexpected calibration_holdout_name values: {unknown_holdouts}")
    missing_holdouts = sorted(expected_holdouts - observed_holdouts)
    if missing_holdouts:
        raise ValueError(f"Calibration guard rows are missing expected holdout provenance: {missing_holdouts}")
    grouped: Dict[ContextPairKey, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    duplicates: List[Dict[str, Any]] = []
    for row in rows:
        schedule = str(row["scheduler_key"])
        if schedule not in set(support_keys):
            continue
        key = context_pair_key(row, pair_on_seed=True)
        if schedule in grouped[key]:
            duplicates.append({"group": list(key), "scheduler_key": schedule})
        copied = dict(row)
        copied["context_id"] = key[3]
        grouped[key][schedule] = copied
    if duplicates:
        raise ValueError(f"Calibration guard rows contain duplicate context/support rows; first={duplicates[:3]}")
    if not grouped:
        raise ValueError("Calibration guard rows contain no requested support rows.")

    cell_context_scores: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    cell_static_scores: Dict[Tuple[str, int], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    cell_oracle_scores: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    cell_oracle_usage: Dict[Tuple[str, int], Counter[str]] = defaultdict(Counter)
    cell_holdout_oracle_scores: Dict[Tuple[str, int], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    cell_holdout_static_scores: Dict[Tuple[str, int], Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    cell_usage: Dict[Tuple[str, int], Counter[str]] = defaultdict(Counter)
    missing: List[Dict[str, Any]] = []
    student.to(device)
    student.eval()
    with torch.no_grad():
        for key, schedule_rows in sorted(grouped.items(), key=lambda item: item[0]):
            _, solver, target_nfe, context_id, _ = key
            missing_support = [schedule for schedule in support_keys if schedule not in schedule_rows]
            if missing_support:
                missing.append({"group": list(key), "missing_schedule_keys": missing_support})
                continue
            if context_id not in context_embeddings:
                raise KeyError(f"Missing calibration guard context embedding for {context_id}.")
            ref = next(iter(schedule_rows.values()))
            setting = setting_features(solver, target_nfe)[None, :].to(device)
            series = torch.tensor([remapped_series_index(ref, series_index_map)], dtype=torch.long, device=device)
            context = torch.tensor(np.asarray(context_embeddings[context_id], dtype=np.float32)[None, :], dtype=torch.float32, device=device)
            probabilities = student.probabilities(setting, series, context)[0].detach().cpu().numpy().astype(np.float64)
            selected_idx = int(np.argmax(probabilities))
            selected_support = str(support_keys[selected_idx])
            cell = (str(solver), int(target_nfe))
            cell_usage[cell][selected_support] += 1
            cell_context_scores[cell].append(float(schedule_rows[selected_support]["u_comp_uniform"]))
            schedule_utilities = {schedule: float(schedule_rows[schedule]["u_comp_uniform"]) for schedule in support_keys}
            oracle_support = sorted(schedule_utilities, key=lambda schedule: (-float(schedule_utilities[schedule]), str(schedule)))[0]
            oracle_score = float(schedule_utilities[oracle_support])
            holdout_name = str(ref.get("calibration_holdout_name", ""))
            cell_oracle_scores[cell].append(oracle_score)
            cell_oracle_usage[cell][oracle_support] += 1
            cell_holdout_oracle_scores[cell][holdout_name].append(oracle_score)
            for schedule in support_keys:
                utility = float(schedule_utilities[schedule])
                cell_static_scores[cell][schedule].append(utility)
                cell_holdout_static_scores[cell][holdout_name][schedule].append(utility)
    if missing:
        raise ValueError(f"Calibration guard rows are missing support schedules for {len(missing)} groups; first={missing[:3]}")

    decisions: List[Dict[str, Any]] = []
    margin_value = float(margin)
    for cell in sorted(cell_static_scores, key=lambda item: (item[0], item[1])):
        solver, target_nfe = cell
        context_values = cell_context_scores.get(cell, [])
        if not context_values:
            raise ValueError(f"Calibration guard has no context-student selections for cell {cell}.")
        context_score = float(np.mean(np.asarray(context_values, dtype=np.float64)))
        static_means = {
            schedule: float(np.mean(np.asarray(values, dtype=np.float64)))
            for schedule, values in sorted(cell_static_scores[cell].items())
            if values
        }
        best_static = sorted(static_means, key=lambda schedule: (-float(static_means[schedule]), str(schedule)))[0]
        best_static_score = float(static_means[best_static])
        oracle_values = cell_oracle_scores.get(cell, [])
        oracle_context_score = float(np.mean(np.asarray(oracle_values, dtype=np.float64))) if oracle_values else None
        oracle_advantage = (
            float(oracle_context_score - best_static_score)
            if oracle_context_score is not None
            else None
        )
        context_capture = (
            float((context_score - best_static_score) / oracle_advantage)
            if oracle_advantage is not None and oracle_advantage > 1e-12
            else None
        )
        oracle_by_holdout: Dict[str, Dict[str, Any]] = {}
        for holdout_name in sorted(cell_holdout_oracle_scores.get(cell, {})):
            holdout_oracle_values = cell_holdout_oracle_scores[cell][holdout_name]
            holdout_static_means = {
                schedule: float(np.mean(np.asarray(values, dtype=np.float64)))
                for schedule, values in sorted(cell_holdout_static_scores[cell][holdout_name].items())
                if values
            }
            holdout_best_static = sorted(
                holdout_static_means,
                key=lambda schedule: (-float(holdout_static_means[schedule]), str(schedule)),
            )[0]
            holdout_best_score = float(holdout_static_means[holdout_best_static])
            holdout_oracle_score = float(np.mean(np.asarray(holdout_oracle_values, dtype=np.float64)))
            oracle_by_holdout[holdout_name] = {
                "oracle_context_score": holdout_oracle_score,
                "best_static_support_schedule_key": str(holdout_best_static),
                "best_static_score": holdout_best_score,
                "oracle_context_advantage_vs_best_static": float(holdout_oracle_score - holdout_best_score),
                "context_group_count": int(len(holdout_oracle_values)),
            }
        deploy_context = bool(context_score >= best_static_score + margin_value)
        decisions.append(
            {
                "solver_key": str(solver),
                "target_nfe": int(target_nfe),
                "deployed_mode": "context_student" if deploy_context else "static_support",
                "best_static_support_schedule_key": str(best_static),
                "fallback_schedule_key": "" if deploy_context else str(best_static),
                "context_student_score": float(context_score),
                "best_static_score": float(best_static_score),
                "score_margin": float(context_score - best_static_score),
                "oracle_context_score": oracle_context_score,
                "oracle_context_advantage_vs_best_static": oracle_advantage,
                "context_student_oracle_capture_fraction": context_capture,
                "oracle_support_usage": {schedule: int(cell_oracle_usage[cell].get(schedule, 0)) for schedule in support_keys},
                "oracle_advantage_by_holdout": oracle_by_holdout,
                "required_margin": float(margin_value),
                "context_group_count": int(len(context_values)),
                "static_support_scores": static_means,
                "student_argmax_support_usage": {schedule: int(cell_usage[cell].get(schedule, 0)) for schedule in support_keys},
            }
        )
    source_context_ids = sorted({context_id_from_row(row) for row in rows})
    source_split_phases = sorted({str(row.get("split_phase", row.get("split", ""))) for row in rows})
    source_row_fingerprints = [
        {
            "dataset": str(row.get("dataset", row.get("dataset_key", ""))),
            "split_phase": str(row.get("split_phase", row.get("split", ""))),
            "seed": str(row.get("seed", "")),
            "solver_key": str(row.get("solver_key", "")),
            "target_nfe": int(row.get("target_nfe", 0)),
            "scheduler_key": str(row.get("scheduler_key", "")),
            "context_id": context_id_from_row(row),
            "series_key": series_key_from_row(row),
            "calibration_holdout_name": str(row.get("calibration_holdout_name", "")),
            "u_comp_uniform": float(row.get("u_comp_uniform", 0.0)),
        }
        for row in rows
    ]
    sorted_row_fingerprints = sorted(
        source_row_fingerprints,
        key=lambda item: tuple(str(item[key]) for key in sorted(item)),
    )
    source_row_hash = hashlib.sha256(
        json.dumps(sorted_row_fingerprints, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    source_context_ids_hash = hashlib.sha256(
        json.dumps(source_context_ids, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    support_schedule_keys_hash = hashlib.sha256(
        json.dumps(list(support_keys), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload_without_id = {
        "artifact": "calibration_holdout_non_regression_guard",
        "enabled": True,
        "reward_key": "u_comp_uniform",
        "margin": float(margin_value),
        "support_schedule_keys": list(support_keys),
        "support_schedule_keys_hash": support_schedule_keys_hash,
        "source_holdouts": [str(name) for name in source_holdout_names],
        "observed_calibration_holdout_names": sorted(observed_holdouts),
        "source_split_phases": source_split_phases,
        "source_context_count": int(len(source_context_ids)),
        "source_context_ids_hash": source_context_ids_hash,
        "source_row_count": int(len(rows)),
        "source_row_hash": source_row_hash,
        "cell_decisions": decisions,
        "cell_decision_map": {f"{item['solver_key']}/{item['target_nfe']}": dict(item) for item in decisions},
        "locked_test_used_for_selection": False,
        "locked_test_used_for_guard_construction": False,
    }
    guard_table_hash = hashlib.sha256(json.dumps(payload_without_id, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    guard_id = "guard_" + guard_table_hash[:24]
    return {**payload_without_id, "guard_id": guard_id, "guard_table_hash": guard_table_hash}


def read_metric_rows_csv(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


__all__ = [
    "CONTEXT_CONDITIONAL_PROTOCOL",
    "DEFAULT_SUPPORT_SCHEDULE_KEYS",
    "DEFAULT_CONTEXT_CALIBRATION_TOTAL",
    "DEFAULT_SUPPORT_CHOICE_MARGIN",
    "DEFAULT_TEACHER_SELECTION_MIN_PAIRWISE_ACCURACY",
    "DEFAULT_TEACHER_SELECTION_MIN_SPEARMAN",
    "MAX_CONTEXT_CALIBRATION_TOTAL",
    "MIN_CONTEXT_CALIBRATION_TOTAL",
    "CONTEXT_SUPPORT_SCHEDULE_KEYS",
    "ContextScheduleTeacherMLP",
    "ContextSupportStudentMLP",
    "EmbeddingNormalizer",
    "attach_uniform_context_rewards",
    "build_calibration_holdout_non_regression_guard",
    "build_teacher_guided_support_targets",
    "build_series_index_map",
    "context_calibration_train_val_counts",
    "context_id_from_row",
    "context_pair_key",
    "context_teacher_diagnostics",
    "grid_for_schedule",
    "load_context_embedding_table",
    "padded_intervals_from_grid",
    "pairwise_rank_loss",
    "read_metric_rows_csv",
    "recommended_context_calibration_count",
    "remapped_series_index",
    "save_context_embedding_table",
    "sample_context_ids_stratified",
    "schedule_grid_hash",
    "series_key_from_row",
    "split_rows_by_context_holdout",
    "split_rows_by_series_holdout",
    "stable_context_id",
    "support_student_ce_loss",
    "train_context_support_student",
    "train_context_teacher",
    "validate_context_support_schedule_keys",
]
