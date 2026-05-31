#!/usr/bin/env python3
"""Locked paper experiment horizons and non-AR rollout chunk sizes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from genode.data.otflow_medical_constants import SLEEP_EDF_DATASET_KEY

FORECAST_FAMILY = "forecast_extrapolation"
CONDITIONAL_GENERATION_FAMILY = "conditional_generation"


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
        dataset_key="wind_farms_wo_missing",
        benchmark_family=FORECAST_FAMILY,
        display_name="Wind Farms (Monash, W/O Missing)",
        experiment_horizon=1440,
        future_block_len=1440,
        history_len=1440,
        reasoning_axis="physical_time",
        rationale="Minute-level extrapolation uses a one-day long horizon, and the rollout is horizon-wise so the scheduler is applied to one full non-AR conditional solve.",
    ),
    DatasetExperimentSpec(
        dataset_key="san_francisco_traffic",
        benchmark_family=FORECAST_FAMILY,
        display_name="San Francisco Traffic (Monash)",
        experiment_horizon=168,
        future_block_len=168,
        history_len=336,
        reasoning_axis="physical_time",
        rationale="Hourly traffic uses a one-week horizon, and the rollout is horizon-wise to avoid chunk-to-chunk distribution shift in the main schedule comparison.",
    ),
    DatasetExperimentSpec(
        dataset_key="london_smart_meters_wo_missing",
        benchmark_family=FORECAST_FAMILY,
        display_name="London Smart Meters (Monash, W/O Missing)",
        experiment_horizon=336,
        future_block_len=336,
        history_len=672,
        reasoning_axis="physical_time",
        rationale="Half-hourly smart meters use a one-week horizon, with a horizon-wise non-AR rollout so later steps are not influenced by an extra chunking policy.",
    ),
    DatasetExperimentSpec(
        dataset_key="electricity",
        benchmark_family=FORECAST_FAMILY,
        display_name="Electricity (Monash)",
        experiment_horizon=168,
        future_block_len=168,
        history_len=336,
        reasoning_axis="physical_time",
        rationale="Hourly electricity uses a one-week horizon, and the rollout is horizon-wise for a cleaner one-schedule-per-horizon comparison.",
    ),
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
        dataset_key="cryptos",
        benchmark_family=CONDITIONAL_GENERATION_FAMILY,
        display_name="cryptos",
        experiment_horizon=200,
        future_block_len=200,
        history_len=256,
        reasoning_axis="event_count",
        rationale="Conditional generation uses a 200-event horizon with a horizon-wise rollout so the scheduler is evaluated on the full event trajectory rather than on repeated sub-blocks.",
    ),
    DatasetExperimentSpec(
        dataset_key="es_mbp_10",
        benchmark_family=CONDITIONAL_GENERATION_FAMILY,
        display_name="es_mbp_10",
        experiment_horizon=200,
        future_block_len=200,
        history_len=256,
        reasoning_axis="event_count",
        rationale="The more irregular ES benchmark keeps the same 200-event horizon, but the main fairness criterion is horizon-wise rollout so schedule comparisons are made on a single full-horizon solve.",
    ),
    DatasetExperimentSpec(
        dataset_key=SLEEP_EDF_DATASET_KEY,
        benchmark_family=CONDITIONAL_GENERATION_FAMILY,
        display_name="sleep_edf",
        experiment_horizon=3000,
        future_block_len=3000,
        history_len=12000,
        reasoning_axis="physical_time",
        rationale="Sleep-EDF uses a 120-second context and a 30-second stage-conditioned continuation at 100 Hz, keeping the main comparison on a single full-horizon non-AR solve while preserving the locked medical benchmark horizon.",
    ),
)

CANONICAL_FORECAST_PAPER_DATASETS: tuple[str, ...] = tuple(
    spec.dataset_key for spec in PAPER_EXPERIMENT_SPECS if spec.benchmark_family == FORECAST_FAMILY
)
CANONICAL_CONDITIONAL_GENERATION_PAPER_DATASETS: tuple[str, ...] = tuple(
    spec.dataset_key for spec in PAPER_EXPERIMENT_SPECS if spec.benchmark_family == CONDITIONAL_GENERATION_FAMILY
)
CHECKPOINT_READY_FORECAST_DATASETS: tuple[str, ...] = tuple(CANONICAL_FORECAST_PAPER_DATASETS)
CHECKPOINT_READY_CONDITIONAL_GENERATION_DATASETS: tuple[str, ...] = tuple(
    CANONICAL_CONDITIONAL_GENERATION_PAPER_DATASETS
)


def experiment_plan_specs() -> List[DatasetExperimentSpec]:
    return list(PAPER_EXPERIMENT_SPECS)


def experiment_plan_by_key() -> Dict[str, DatasetExperimentSpec]:
    return {spec.dataset_key: spec for spec in PAPER_EXPERIMENT_SPECS}


def canonical_forecast_paper_dataset_keys() -> tuple[str, ...]:
    return tuple(CANONICAL_FORECAST_PAPER_DATASETS)


def canonical_conditional_generation_paper_dataset_keys() -> tuple[str, ...]:
    return tuple(CANONICAL_CONDITIONAL_GENERATION_PAPER_DATASETS)


def checkpoint_ready_forecast_dataset_keys() -> tuple[str, ...]:
    return tuple(CHECKPOINT_READY_FORECAST_DATASETS)


def checkpoint_ready_conditional_generation_dataset_keys() -> tuple[str, ...]:
    return tuple(CHECKPOINT_READY_CONDITIONAL_GENERATION_DATASETS)


def validate_experiment_plan(specs: Iterable[DatasetExperimentSpec] | None = None) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
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
            "horizon_rule": "Use reviewer-facing long horizons in physical time for forecasting and event-count horizons for conditional generation.",
            "chunk_rule": "Use horizon-wise non-AR rollouts in the main experiments, i.e. future_block_len equals the experiment horizon.",
        },
        "datasets": [asdict(spec) for spec in PAPER_EXPERIMENT_SPECS],
        "validation": validation_rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


__all__ = [
    "CONDITIONAL_GENERATION_FAMILY",
    "CANONICAL_FORECAST_PAPER_DATASETS",
    "CANONICAL_CONDITIONAL_GENERATION_PAPER_DATASETS",
    "CHECKPOINT_READY_FORECAST_DATASETS",
    "CHECKPOINT_READY_CONDITIONAL_GENERATION_DATASETS",
    "DatasetExperimentSpec",
    "FORECAST_FAMILY",
    "PAPER_EXPERIMENT_SPECS",
    "canonical_forecast_paper_dataset_keys",
    "canonical_conditional_generation_paper_dataset_keys",
    "checkpoint_ready_forecast_dataset_keys",
    "checkpoint_ready_conditional_generation_dataset_keys",
    "experiment_plan_by_key",
    "experiment_plan_specs",
    "validate_experiment_plan",
    "write_experiment_plan",
]
