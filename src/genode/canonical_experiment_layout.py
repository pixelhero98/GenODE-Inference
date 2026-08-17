from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

from genode.schedule_transfer.reference_clocks import (
    DEFAULT_REFERENCE_CLOCK_KEYS,
    REFERENCE_CLOCK_BASE_KEYS,
    REFERENCE_CLOCK_REVERSED_KEYS,
)
from genode.solver_protocol import CANONICAL_SOLVER_KEYS

CANONICAL_LAYOUT_VERSION = "seen_unseen_nfe_layout"

CANONICAL_SEEN_NFES: Tuple[int, ...] = (4, 8, 12, 16)
CANONICAL_UNSEEN_NFES: Tuple[int, ...] = (6, 10, 14, 20)
CANONICAL_CHECKPOINT_STEPS: Tuple[int, ...] = (4000, 8000, 12000, 16000, 20000)
CANONICAL_CONTEXT_SAMPLE_COUNT = 188
CANONICAL_UNSEEN_TARGET_WEIGHT = 0.25

NFE_ROLE_SEEN = "seen"
NFE_ROLE_UNSEEN = "unseen"
NFE_ROLES: Tuple[str, ...] = (NFE_ROLE_SEEN, NFE_ROLE_UNSEEN)

SCENARIO_FAMILY_FORECAST = "temporal_extrapolation"
SCENARIO_FAMILY_CONDITIONAL_GENERATION = "temporal_conditional_generation"
SCENARIO_FAMILY_MOLECULE = "molecule_3d_coordinate_generation"

FORECAST_SCENARIO_KEYS: Tuple[str, ...] = (
    "solar_energy_10m",
    "traffic_hourly",
    "weather_daily",
)
CONDITIONAL_GENERATION_SCENARIO_KEYS: Tuple[str, ...] = (
    "cryptos",
    "lobster_synthetic",
    "long_term_st",
)
MOLECULE_SCENARIO_KEYS: Tuple[str, ...] = (
    "molecule_3d_set1",
    "molecule_3d_set2",
    "molecule_3d_set3",
)
CANONICAL_SCENARIO_KEYS: Tuple[str, ...] = (
    *FORECAST_SCENARIO_KEYS,
    *CONDITIONAL_GENERATION_SCENARIO_KEYS,
    *MOLECULE_SCENARIO_KEYS,
)

PHYSICAL_SCHEDULE_KEYS: Tuple[str, ...] = REFERENCE_CLOCK_BASE_KEYS
REVERSED_SCHEDULE_KEYS: Tuple[str, ...] = REFERENCE_CLOCK_REVERSED_KEYS
# Density-averaged and SER clocks remain available only through explicit, non-canonical inputs.
AVERAGED_REVERSED_SCHEDULE_KEYS: Tuple[str, ...] = ()
CANONICAL_SUPERVISION_SCHEDULE_KEYS: Tuple[str, ...] = DEFAULT_REFERENCE_CLOCK_KEYS

REVERSED_SCHEDULE_BASE: Mapping[str, str] = {key: key.removesuffix("_reversed") for key in REVERSED_SCHEDULE_KEYS}
AVERAGED_SCHEDULE_COMPONENTS: Mapping[str, Tuple[str, str]] = {}

SCHEDULE_FAMILY_PHYSICAL = "physical"
SCHEDULE_FAMILY_REVERSED = "reversed"
SCHEDULE_FAMILY_AVERAGED_REVERSED = "averaged_reversed"

STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT = "seen_only_zero_shot"
STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGETS = "seen_plus_unseen_targets"
STUDENT_TRAINING_MODES: Tuple[str, ...] = (
    STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT,
    STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGETS,
)


@dataclass(frozen=True)
class ScenarioSpec:
    key: str
    family: str
    public_dataset_key: str


def canonical_scenario_specs() -> Tuple[ScenarioSpec, ...]:
    return tuple(
        [
            ScenarioSpec(key=key, family=SCENARIO_FAMILY_FORECAST, public_dataset_key=key)
            for key in FORECAST_SCENARIO_KEYS
        ]
        + [
            ScenarioSpec(key=key, family=SCENARIO_FAMILY_CONDITIONAL_GENERATION, public_dataset_key=key)
            for key in CONDITIONAL_GENERATION_SCENARIO_KEYS
        ]
        + [
            ScenarioSpec(key=key, family=SCENARIO_FAMILY_MOLECULE, public_dataset_key=key)
            for key in MOLECULE_SCENARIO_KEYS
        ]
    )


def scenario_family_for_key(scenario_key: str) -> str:
    key = str(scenario_key)
    for spec in canonical_scenario_specs():
        if spec.key == key:
            return spec.family
    raise KeyError(f"Unknown canonical scenario key: {scenario_key!r}")


def canonical_nfes_for_role(nfe_role: str) -> Tuple[int, ...]:
    role = str(nfe_role).strip().lower()
    if role == NFE_ROLE_SEEN:
        return CANONICAL_SEEN_NFES
    if role == NFE_ROLE_UNSEEN:
        return CANONICAL_UNSEEN_NFES
    raise ValueError(f"Unknown nfe_role={nfe_role!r}; expected one of {NFE_ROLES}.")


def schedule_family_for_key(schedule_key: str) -> str:
    key = str(schedule_key).strip()
    if key in PHYSICAL_SCHEDULE_KEYS:
        return SCHEDULE_FAMILY_PHYSICAL
    if key in REVERSED_SCHEDULE_KEYS:
        return SCHEDULE_FAMILY_REVERSED
    if key in AVERAGED_REVERSED_SCHEDULE_KEYS:
        return SCHEDULE_FAMILY_AVERAGED_REVERSED
    return "generated"


def density_source_key_for_schedule(schedule_key: str) -> str:
    key = str(schedule_key).strip()
    if key in AVERAGED_SCHEDULE_COMPONENTS:
        left, right = AVERAGED_SCHEDULE_COMPONENTS[key]
        return f"{left}+{right}"
    if key in REVERSED_SCHEDULE_BASE:
        return REVERSED_SCHEDULE_BASE[key]
    return key


def canonical_layout_summary() -> Dict[str, object]:
    return {
        "canonical_layout_version": CANONICAL_LAYOUT_VERSION,
        "seen_nfes": list(CANONICAL_SEEN_NFES),
        "unseen_nfes": list(CANONICAL_UNSEEN_NFES),
        "checkpoint_steps": list(CANONICAL_CHECKPOINT_STEPS),
        "context_sample_count": int(CANONICAL_CONTEXT_SAMPLE_COUNT),
        "solver_keys": list(CANONICAL_SOLVER_KEYS),
        "scenario_keys": list(CANONICAL_SCENARIO_KEYS),
        "physical_schedule_keys": list(PHYSICAL_SCHEDULE_KEYS),
        "reversed_schedule_keys": list(REVERSED_SCHEDULE_KEYS),
        "averaged_reversed_schedule_keys": list(AVERAGED_REVERSED_SCHEDULE_KEYS),
        "supervision_schedule_keys": list(CANONICAL_SUPERVISION_SCHEDULE_KEYS),
    }


__all__ = [
    "AVERAGED_REVERSED_SCHEDULE_KEYS",
    "AVERAGED_SCHEDULE_COMPONENTS",
    "CANONICAL_CHECKPOINT_STEPS",
    "CANONICAL_CONTEXT_SAMPLE_COUNT",
    "CANONICAL_LAYOUT_VERSION",
    "CANONICAL_UNSEEN_TARGET_WEIGHT",
    "CANONICAL_SCENARIO_KEYS",
    "CANONICAL_SEEN_NFES",
    "CANONICAL_SOLVER_KEYS",
    "CANONICAL_SUPERVISION_SCHEDULE_KEYS",
    "CANONICAL_UNSEEN_NFES",
    "CONDITIONAL_GENERATION_SCENARIO_KEYS",
    "FORECAST_SCENARIO_KEYS",
    "MOLECULE_SCENARIO_KEYS",
    "NFE_ROLE_SEEN",
    "NFE_ROLE_UNSEEN",
    "NFE_ROLES",
    "PHYSICAL_SCHEDULE_KEYS",
    "REVERSED_SCHEDULE_BASE",
    "REVERSED_SCHEDULE_KEYS",
    "SCENARIO_FAMILY_CONDITIONAL_GENERATION",
    "SCENARIO_FAMILY_FORECAST",
    "SCENARIO_FAMILY_MOLECULE",
    "SCHEDULE_FAMILY_AVERAGED_REVERSED",
    "SCHEDULE_FAMILY_PHYSICAL",
    "SCHEDULE_FAMILY_REVERSED",
    "STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT",
    "STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGETS",
    "STUDENT_TRAINING_MODES",
    "ScenarioSpec",
    "canonical_layout_summary",
    "canonical_nfes_for_role",
    "canonical_scenario_specs",
    "density_source_key_for_schedule",
    "scenario_family_for_key",
    "schedule_family_for_key",
]
