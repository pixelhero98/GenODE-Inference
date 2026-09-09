"""Terminal metric profiles for retained forecasting and molecular tasks."""

from __future__ import annotations

from dataclasses import dataclass

from genode.canonical_experiment_layout import (
    SCENARIO_FAMILY_FORECAST,
    SCENARIO_FAMILY_MOLECULE,
    scenario_family_for_key,
)

METRIC_DIRECTION_LOWER = "lower"


@dataclass(frozen=True)
class MetricObjectiveSpec:
    metric_key: str
    utility_key: str
    direction: str
    weight: float = 1.0
    aliases: tuple[str, ...] = ()


FORECAST_METRIC_SPECS: tuple[MetricObjectiveSpec, ...] = (
    MetricObjectiveSpec("crps", "u_crps_uniform", METRIC_DIRECTION_LOWER, 0.5, aliases=("forecast_crps",)),
    MetricObjectiveSpec("mase", "u_mase_uniform", METRIC_DIRECTION_LOWER, 0.5, aliases=("forecast_mase",)),
)

MOLECULE_METRIC_SPECS = (
    MetricObjectiveSpec("molecule_energy_score", "u_energy_score_uniform", METRIC_DIRECTION_LOWER, 1.0),
)
OBJECTIVE_SPECS_BY_FAMILY = {
    SCENARIO_FAMILY_FORECAST: FORECAST_METRIC_SPECS,
    SCENARIO_FAMILY_MOLECULE: MOLECULE_METRIC_SPECS,
}


def teacher_objective_specs_for_scenario(scenario_key):
    return OBJECTIVE_SPECS_BY_FAMILY[scenario_family_for_key(scenario_key)]


def teacher_metric_profile_for_scenario(scenario_key):
    specs = teacher_objective_specs_for_scenario(scenario_key)
    return {
        "scenario_key": scenario_key,
        "benchmark_family": scenario_family_for_key(scenario_key),
        "target_metric_keys": [s.metric_key for s in specs],
        "target_utility_keys": [s.utility_key for s in specs],
        "target_weights": {s.utility_key: s.weight for s in specs},
        "diagnostic_metric_keys": [],
        "diagnostic_utility_keys": [],
    }
