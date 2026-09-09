"""Paired terminal measurements and calibration frozen before policy fitting."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

TASK_METRICS = {
    "solar_energy_10m": ("crps", "mase"),
    "traffic_hourly": ("crps", "mase"),
    "weather_daily": ("crps", "mase"),
    "molecule_3d_set1": ("energy_score",),
    "molecule_3d_set2": ("energy_score",),
    "molecule_3d_set3": ("energy_score",),
    "cifar10": ("kid",),
    "imagenet64": ("kid",),
    "sana": ("preference", "alignment"),
    "sd15": ("preference", "alignment"),
}
LOG_METRICS = frozenset(("crps", "mase", "energy_score"))
FIT_SPLITS = frozenset(("train", "calibration"))


def _balanced_std(values: np.ndarray, nfes: np.ndarray) -> np.ndarray:
    """Population standard deviation with equal total weight per observed NFE."""
    weights = np.zeros(len(nfes), dtype=np.float64)
    unique = np.unique(nfes)
    for nfe in unique:
        mask = nfes == nfe
        weights[mask] = 1 / (len(unique) * mask.sum())
    mean = np.sum(values * weights[:, None], axis=0)
    return np.sqrt(np.sum((values - mean) ** 2 * weights[:, None], axis=0))


def _paired_cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair before averaging repeats. Never silently intersect seed panels."""
    if not rows:
        raise ValueError("Reward evidence is empty.")
    panel: dict[tuple, dict[str, dict]] = defaultdict(dict)
    required = (
        "task",
        "backbone",
        "solver",
        "nfe",
        "context_id",
        "split",
        "seed",
        "ensemble_size",
        "reference_id",
        "measurement_protocol",
        "schedule_key",
        "metrics",
    )
    for row in rows:
        missing = set(required) - row.keys()
        if missing:
            raise ValueError(f"Measurement is missing pairing fields: {sorted(missing)}")
        if row["task"] not in TASK_METRICS:
            raise ValueError(f"Unsupported reward task: {row['task']!r}")
        if row["split"] not in (*FIT_SPLITS, "validation", "test"):
            raise ValueError("Measurements require an explicit train/calibration/validation/test split.")
        for key in ("nfe", "ensemble_size"):
            if isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] < 1:
                raise ValueError(f"{key} must be a positive integer.")
        if isinstance(row["seed"], bool) or not isinstance(row["seed"], int):
            raise ValueError("seed must be an integer.")
        for key in ("backbone", "context_id", "reference_id", "measurement_protocol", "schedule_key"):
            if not isinstance(row[key], str) or not row[key]:
                raise ValueError(f"{key} must be a nonempty identity.")
        values = np.array([row["metrics"][key] for key in TASK_METRICS[row["task"]]], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Terminal metrics must be finite and complete.")
        if "reward_metrics" in row:
            if row["task"] != "imagenet64" or row["split"] not in FIT_SPLITS or "reward_estimator" not in row:
                raise ValueError("Only calibrated ImageNet training evidence may carry shrunk reward metrics.")
            if set(row["reward_metrics"]) != {"kid"} or not np.isfinite(row["reward_metrics"]["kid"]):
                raise ValueError("Invalid shrunk KID evidence.")
        if any(row["metrics"][key] < 0 for key in TASK_METRICS[row["task"]] if key in LOG_METRICS):
            raise ValueError("Error metrics must be nonnegative.")
        if "energy_score" in TASK_METRICS[row["task"]] and row["ensemble_size"] < 2:
            raise ValueError("Fair molecular energy score requires at least two ensemble members.")
        key = tuple(row[k] for k in ("task", "backbone", "solver", "nfe", "context_id", "split", "seed"))
        if row["schedule_key"] in panel[key]:
            raise ValueError("Duplicate schedule measurement in a paired seed panel.")
        panel[key][row["schedule_key"]] = row
    groups: dict[tuple, list[dict]] = defaultdict(list)
    supports: dict[tuple, set[str]] = {}
    for key, schedules in panel.items():
        if "uniform" not in schedules:
            raise ValueError("Every seed panel requires its measured uniform anchor.")
        group = key[:-1]
        support = set(schedules)
        if group in supports and supports[group] != support:
            raise ValueError("Candidate and anchor seed panels differ; partial pairing is not permitted.")
        supports[group] = support
        anchor = schedules["uniform"]
        for row in schedules.values():
            for field in ("ensemble_size", "reference_id", "measurement_protocol"):
                if row[field] != anchor[field]:
                    raise ValueError(f"Candidate and uniform have different {field}.")
            groups[(*group, row["schedule_key"])].append(row)
    cells = []
    for repeats in groups.values():
        first = repeats[0]
        keys = TASK_METRICS[first["task"]]
        cell = {**first, "metrics": {k: float(np.mean([r["metrics"][k] for r in repeats])) for k in keys}}
        if any("reward_metrics" in r for r in repeats):
            if not all("reward_metrics" in r for r in repeats):
                raise ValueError("Repeated KID measurements have inconsistent shrinkage protocols.")
            cell["reward_metrics"] = {k: float(np.mean([r["reward_metrics"][k] for r in repeats])) for k in keys}
        cell["seeds"] = sorted(r["seed"] for r in repeats)
        cell["reference_ids"] = {str(r["seed"]): r["reference_id"] for r in repeats}
        cell.pop("seed")
        # Density and measurement semantics may not change between repeats.
        for r in repeats[1:]:
            for field in ("density_mass", "time_grid", "measurement_protocol", "ensemble_size"):
                if r.get(field) != first.get(field):
                    raise ValueError(f"Repeated measurements disagree on {field}.")
        cells.append(cell)
    lookup = {
        (r["task"], r["backbone"], r["solver"], r["nfe"], r["context_id"], r["split"]): r
        for r in cells
        if r["schedule_key"] == "uniform"
    }
    for cell in cells:
        anchor = lookup[tuple(cell[k] for k in ("task", "backbone", "solver", "nfe", "context_id", "split"))]
        cell["anchor_metrics"] = anchor["metrics"]
        cell["anchor_reward_metrics"] = anchor.get("reward_metrics", anchor["metrics"])
    return cells


@dataclass(frozen=True)
class RewardCalibration:
    task: str
    backbone: str
    solver: str
    metric_keys: tuple[str, ...]
    floors: tuple[float, ...]
    component_scales: tuple[float, ...]
    reward_scale: float
    calibration_contexts: tuple[str, ...]
    calibration_nfes: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.task not in TASK_METRICS or tuple(self.metric_keys) != TASK_METRICS[self.task]:
            raise ValueError("Calibration metric profile does not match its task.")
        count = len(self.metric_keys)
        if len(self.floors) != count or len(self.component_scales) != count:
            raise ValueError("Calibration vectors have inconsistent widths.")
        if not np.isfinite([*self.floors, *self.component_scales, self.reward_scale]).all():
            raise ValueError("Calibration scales must be finite.")
        if min(self.component_scales) <= 0 or self.reward_scale <= 1e-12:
            raise ValueError("Degenerate reward calibration.")
        if any(f <= 0 for k, f in zip(self.metric_keys, self.floors, strict=True) if k in LOG_METRICS):
            raise ValueError("Positive-error calibration requires positive frozen numerical floors.")

    def vector(self, cell: dict, *, normalize: bool = True) -> np.ndarray:
        if (cell["task"], cell["backbone"], cell["solver"]) != (self.task, self.backbone, self.solver):
            raise ValueError("Reward calibration task/backbone/solver mismatch.")
        values = []
        for key, floor, scale in zip(self.metric_keys, self.floors, self.component_scales, strict=True):
            candidate = cell.get("reward_metrics", cell["metrics"])[key]
            anchor = cell.get("anchor_reward_metrics", cell["anchor_metrics"])[key]
            if key in LOG_METRICS:
                value = np.log((anchor + floor) / (candidate + floor))
            else:
                value = ((anchor - candidate) if key == "kid" else (candidate - anchor)) / scale
            values.append(value)
        result = np.asarray(values, dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite paired reward.")
        return result / self.reward_scale if normalize else result

    def to_payload(self) -> dict:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict) -> RewardCalibration:
        values = dict(payload)
        for name in ("metric_keys", "floors", "component_scales", "calibration_contexts", "calibration_nfes"):
            values[name] = tuple(values[name])
        return cls(**values)


def calibrate_rewards(rows: list[dict], *, component_calibration_rows: list[dict] | None = None) -> RewardCalibration:
    if any(r.get("split") not in FIT_SPLITS for r in rows):
        raise ValueError("Reward calibration may only consume train/calibration measurements.")
    cells = _paired_cells(rows)
    scopes = {(r["task"], r["backbone"], r["solver"]) for r in cells}
    if len(scopes) != 1:
        raise ValueError("Calibrate one task/backbone/solver at a time.")
    task, backbone, solver = scopes.pop()
    metrics = TASK_METRICS[task]
    anchors = [r for r in cells if r["schedule_key"] == "uniform"]
    candidates = [r for r in cells if r["schedule_key"] != "uniform"]
    if not candidates:
        raise ValueError("Calibration requires non-uniform candidates.")
    floors = []
    for key in metrics:
        positive = [r["metrics"][key] for r in anchors if r["metrics"][key] > 0]
        if key in LOG_METRICS and not positive:
            raise ValueError(f"Degenerate uniform calibration for {key}.")
        floors.append(float(1e-6 * np.median(positive)) if key in LOG_METRICS else 0.0)
    nfes = np.array([r["nfe"] for r in candidates])
    scales = np.ones(len(metrics))
    component_cells = cells
    if component_calibration_rows is not None:
        if task not in ("sana", "sd15") or any(r.get("split") not in FIT_SPLITS for r in component_calibration_rows):
            raise ValueError("Separate component calibration requires text-to-image pilot evidence.")
        component_cells = _paired_cells(component_calibration_rows)
        if any((r["task"], r["backbone"], r["solver"]) != (task, backbone, solver) for r in component_cells):
            raise ValueError("Pilot component calibration scope differs from training evidence.")
    if task in ("sana", "sd15"):
        pilot = [r for r in component_cells if r["schedule_key"] != "uniform"]
        if not pilot:
            raise ValueError("Component calibration requires non-uniform pilot candidates.")
        differences = np.array([[r["metrics"][k] - r["anchor_metrics"][k] for k in metrics] for r in pilot])
        scales = _balanced_std(differences, np.array([r["nfe"] for r in pilot]))
        if np.any(scales <= 1e-12):
            raise ValueError("Degenerate text-to-image component calibration.")
    common = {
        "task": task,
        "backbone": backbone,
        "solver": solver,
        "metric_keys": metrics,
        "floors": tuple(floors),
        "component_scales": tuple(float(x) for x in scales),
        "calibration_contexts": tuple(sorted({r["context_id"] for r in cells + component_cells})),
        "calibration_nfes": tuple(int(x) for x in np.unique(nfes)),
    }
    provisional = RewardCalibration(**common, reward_scale=1.0)
    scalar = np.array([[provisional.vector(r, normalize=False).mean()] for r in candidates])
    return RewardCalibration(**common, reward_scale=float(_balanced_std(scalar, nfes)[0]))


def construct_rewards(rows: list[dict], calibration: RewardCalibration) -> list[dict]:
    cells = _paired_cells(rows)
    for cell in cells:
        vector = calibration.vector(cell)
        cell["reward_vector"] = vector.tolist()
        cell["reward"] = float(vector.mean())
    return cells
