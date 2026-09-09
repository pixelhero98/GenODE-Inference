"""Locked paper experiment horizons and non-AR rollout chunk sizes."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

FORECAST_FAMILY = "temporal_extrapolation"


@dataclass(frozen=True)
class DatasetExperimentSpec:
    dataset_key: str
    benchmark_family: str
    display_name: str
    experiment_horizon: int
    future_block_len: int
    history_len: int
    reasoning_axis: str
    rationale: str


PAPER_EXPERIMENT_SPECS: tuple[DatasetExperimentSpec, ...] = (
    DatasetExperimentSpec(
        dataset_key="solar_energy_10m",
        benchmark_family=FORECAST_FAMILY,
        display_name="Solar Energy (Monash, 10m)",
        experiment_horizon=1008,
        future_block_len=1008,
        history_len=1008,
        reasoning_axis="physical_time",
        rationale="10-minute solar uses a one-week horizon, and the rollout is horizon-wise so the non-AR comparison is not confounded by intermediate block stitching.",
    ),
    DatasetExperimentSpec(
        dataset_key="traffic_hourly",
        benchmark_family=FORECAST_FAMILY,
        display_name="Traffic Hourly (Monash)",
        experiment_horizon=168,
        future_block_len=168,
        history_len=336,
        reasoning_axis="physical_time",
        rationale="Hourly traffic uses a one-week horizon, and the rollout is horizon-wise to avoid chunk-to-chunk distribution shift in the main schedule comparison.",
    ),
    DatasetExperimentSpec(
        dataset_key="weather_daily",
        benchmark_family=FORECAST_FAMILY,
        display_name="Weather Daily (Monash)",
        experiment_horizon=30,
        future_block_len=30,
        history_len=120,
        reasoning_axis="physical_time",
        rationale="Daily weather uses the official 30-day horizon with a 120-day context, keeping the schedule comparison horizon-wise.",
    ),
)
EXPERIMENTAL_EXPERIMENT_SPECS: tuple[DatasetExperimentSpec, ...] = ()
SUPPORTED_EXPERIMENT_SPECS: tuple[DatasetExperimentSpec, ...] = PAPER_EXPERIMENT_SPECS + EXPERIMENTAL_EXPERIMENT_SPECS
CANONICAL_FORECAST_PAPER_DATASETS: tuple[str, ...] = tuple(
    spec.dataset_key for spec in PAPER_EXPERIMENT_SPECS if spec.benchmark_family == FORECAST_FAMILY
)
CHECKPOINT_READY_FORECAST_DATASETS: tuple[str, ...] = tuple(CANONICAL_FORECAST_PAPER_DATASETS)


def experiment_plan_specs() -> list[DatasetExperimentSpec]:
    return list(PAPER_EXPERIMENT_SPECS)


def experiment_plan_by_key() -> dict[str, DatasetExperimentSpec]:
    return {spec.dataset_key: spec for spec in SUPPORTED_EXPERIMENT_SPECS}


def canonical_forecast_paper_dataset_keys() -> tuple[str, ...]:
    return tuple(CANONICAL_FORECAST_PAPER_DATASETS)


def checkpoint_ready_forecast_dataset_keys() -> tuple[str, ...]:
    return tuple(CHECKPOINT_READY_FORECAST_DATASETS)


def validate_experiment_plan(specs: Iterable[DatasetExperimentSpec] | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in PAPER_EXPERIMENT_SPECS if specs is None else list(specs):
        divides = int(spec.experiment_horizon) % int(spec.future_block_len) == 0
        rows.append(
            {
                "dataset_key": spec.dataset_key,
                "benchmark_family": spec.benchmark_family,
                "experiment_horizon": int(spec.experiment_horizon),
                "future_block_len": int(spec.future_block_len),
                "history_len": int(spec.history_len),
                "n_chunks_per_rollout": int(spec.experiment_horizon) // int(spec.future_block_len) if divides else None,
                "future_block_divides_horizon": bool(divides),
            }
        )
    return rows


def write_experiment_plan(out_root: str | Path) -> Mapping[str, object]:
    out_path = Path(out_root).resolve() / "experiment_plan.json"
    validation_rows = validate_experiment_plan()
    payload = {
        "locked": True,
        "selection_policy": {
            "horizon_rule": "Use reviewer-facing long horizons in physical time for forecasting.",
            "chunk_rule": "Use horizon-wise non-AR rollouts in the main experiments, i.e. future_block_len equals the experiment horizon.",
        },
        "datasets": [asdict(spec) for spec in PAPER_EXPERIMENT_SPECS],
        "validation": validation_rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


__all__ = [
    "CANONICAL_FORECAST_PAPER_DATASETS",
    "CHECKPOINT_READY_FORECAST_DATASETS",
    "EXPERIMENTAL_EXPERIMENT_SPECS",
    "SUPPORTED_EXPERIMENT_SPECS",
    "DatasetExperimentSpec",
    "FORECAST_FAMILY",
    "PAPER_EXPERIMENT_SPECS",
    "canonical_forecast_paper_dataset_keys",
    "checkpoint_ready_forecast_dataset_keys",
    "experiment_plan_by_key",
    "experiment_plan_specs",
    "validate_experiment_plan",
    "write_experiment_plan",
]
