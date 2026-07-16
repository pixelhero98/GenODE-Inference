#!/usr/bin/env python3
"""Locked paper experiment horizons and non-AR rollout chunk sizes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from genode.data.otflow_medical_constants import LONG_TERM_ST_DATASET_KEY

FORECAST_FAMILY = "temporal_extrapolation"
CONDITIONAL_GENERATION_FAMILY = "temporal_conditional_generation"


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
    DatasetExperimentSpec(
        dataset_key="cryptos",
        benchmark_family=CONDITIONAL_GENERATION_FAMILY,
        display_name="cryptos",
        experiment_horizon=128,
        future_block_len=128,
        history_len=256,
        reasoning_axis="event_count",
        rationale="Conditional generation uses a horizon set to half the 256-event history length, with a horizon-wise rollout so the scheduler is evaluated on the full event trajectory rather than on repeated sub-blocks.",
    ),
    DatasetExperimentSpec(
        dataset_key="lobster_synthetic",
        benchmark_family=CONDITIONAL_GENERATION_FAMILY,
        display_name="lobster_synthetic",
        experiment_horizon=128,
        future_block_len=128,
        history_len=256,
        reasoning_axis="event_count",
        rationale="LOBSTER-calibrated synthetic order-book continuation uses the same event-count context and horizon as cryptos, generated from the public lobiflow profile.",
    ),
    DatasetExperimentSpec(
        dataset_key=LONG_TERM_ST_DATASET_KEY,
        benchmark_family=CONDITIONAL_GENERATION_FAMILY,
        display_name="long_term_st",
        experiment_horizon=3000,
        future_block_len=3000,
        history_len=12000,
        reasoning_axis="physical_time",
        rationale="Long-Term ST uses a context-only ECG continuation task after strict WFDB validation and downsampling from 250 Hz to 100 Hz.",
    ),
)

EXPERIMENTAL_EXPERIMENT_SPECS: tuple[DatasetExperimentSpec, ...] = ()

SUPPORTED_EXPERIMENT_SPECS: tuple[DatasetExperimentSpec, ...] = (
    PAPER_EXPERIMENT_SPECS + EXPERIMENTAL_EXPERIMENT_SPECS
)

PAPER_FORECAST_DATASETS: tuple[str, ...] = tuple(
    spec.dataset_key for spec in PAPER_EXPERIMENT_SPECS if spec.benchmark_family == FORECAST_FAMILY
)
PAPER_CONDITIONAL_GENERATION_DATASETS: tuple[str, ...] = tuple(
    spec.dataset_key for spec in PAPER_EXPERIMENT_SPECS if spec.benchmark_family == CONDITIONAL_GENERATION_FAMILY
)
SUPPORTED_CONDITIONAL_GENERATION_DATASETS: tuple[str, ...] = tuple(
    spec.dataset_key for spec in SUPPORTED_EXPERIMENT_SPECS if spec.benchmark_family == CONDITIONAL_GENERATION_FAMILY
)


def experiment_plan_by_key() -> Dict[str, DatasetExperimentSpec]:
    return {spec.dataset_key: spec for spec in SUPPORTED_EXPERIMENT_SPECS}


def forecast_dataset_keys() -> tuple[str, ...]:
    return tuple(PAPER_FORECAST_DATASETS)


def conditional_generation_dataset_keys() -> tuple[str, ...]:
    return tuple(PAPER_CONDITIONAL_GENERATION_DATASETS)


__all__ = [
    "CONDITIONAL_GENERATION_FAMILY",
    "PAPER_FORECAST_DATASETS",
    "PAPER_CONDITIONAL_GENERATION_DATASETS",
    "EXPERIMENTAL_EXPERIMENT_SPECS",
    "SUPPORTED_CONDITIONAL_GENERATION_DATASETS",
    "SUPPORTED_EXPERIMENT_SPECS",
    "DatasetExperimentSpec",
    "FORECAST_FAMILY",
    "PAPER_EXPERIMENT_SPECS",
    "forecast_dataset_keys",
    "conditional_generation_dataset_keys",
    "experiment_plan_by_key",
]
