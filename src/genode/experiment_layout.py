from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

EXPERIMENT_LAYOUT_ID = "seen_unseen_nfe_layout"

REFERENCE_SEEN_NFES: Tuple[int, ...] = (4, 8, 12, 16)
REFERENCE_UNSEEN_NFES: Tuple[int, ...] = (6, 10, 14, 20)
REFERENCE_CHECKPOINT_STEPS: Tuple[int, ...] = (4000, 8000, 12000, 16000, 20000)
TRAIN_TUNING_CONTEXT_SAMPLE_COUNT = 188
LOCKED_TEST_PREVIEW_CONTEXTS = 512
REFERENCE_UNSEEN_TARGET_WEIGHT = 0.25

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
REFERENCE_SCENARIO_KEYS: Tuple[str, ...] = (
    *FORECAST_SCENARIO_KEYS,
    *CONDITIONAL_GENERATION_SCENARIO_KEYS,
    *MOLECULE_SCENARIO_KEYS,
)

PHYSICAL_SCHEDULE_KEYS: Tuple[str, ...] = (
    "uniform",
    "late_power_3",
    "flowts_power_sampling",
    "ays",
    "gits",
    "ots",
    "ser_ptg_local_defect_eta005",
)
REVERSED_SCHEDULE_KEYS: Tuple[str, ...] = (
    "late_power_3_reversed",
    "flowts_power_sampling_reversed",
    "ays_reversed",
    "gits_reversed",
    "ots_reversed",
    "ser_ptg_local_defect_eta005_reversed",
)
AVERAGED_REVERSED_SCHEDULE_KEYS: Tuple[str, ...] = (
    "late_power_3_avg_reversed",
    "flowts_power_sampling_avg_reversed",
    "ays_avg_reversed",
    "gits_avg_reversed",
    "ots_avg_reversed",
    "ser_ptg_local_defect_eta005_avg_reversed",
)
REFERENCE_SUPERVISION_SCHEDULE_KEYS: Tuple[str, ...] = (
    *PHYSICAL_SCHEDULE_KEYS,
    *REVERSED_SCHEDULE_KEYS,
    *AVERAGED_REVERSED_SCHEDULE_KEYS,
)

REVERSED_SCHEDULE_BASE: Mapping[str, str] = {
    "late_power_3_reversed": "late_power_3",
    "flowts_power_sampling_reversed": "flowts_power_sampling",
    "ays_reversed": "ays",
    "gits_reversed": "gits",
    "ots_reversed": "ots",
    "ser_ptg_local_defect_eta005_reversed": "ser_ptg_local_defect_eta005",
}
AVERAGED_SCHEDULE_COMPONENTS: Mapping[str, Tuple[str, str]] = {
    "late_power_3_avg_reversed": ("late_power_3", "late_power_3_reversed"),
    "flowts_power_sampling_avg_reversed": ("flowts_power_sampling", "flowts_power_sampling_reversed"),
    "ays_avg_reversed": ("ays", "ays_reversed"),
    "gits_avg_reversed": ("gits", "gits_reversed"),
    "ots_avg_reversed": ("ots", "ots_reversed"),
    "ser_ptg_local_defect_eta005_avg_reversed": (
        "ser_ptg_local_defect_eta005",
        "ser_ptg_local_defect_eta005_reversed",
    ),
}

SCHEDULE_FAMILY_PHYSICAL = "physical"
SCHEDULE_FAMILY_REVERSED = "reversed"
SCHEDULE_FAMILY_AVERAGED_REVERSED = "averaged_reversed"

STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT = "seen_only_zero_shot"
STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGET = "seen_plus_unseen_target"
STUDENT_TRAINING_MODES: Tuple[str, ...] = (
    STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT,
    STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGET,
)


@dataclass(frozen=True)
class ScenarioSpec:
    key: str
    family: str
    public_dataset_key: str


def scenario_specs() -> Tuple[ScenarioSpec, ...]:
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
    for spec in scenario_specs():
        if spec.key == key:
            return spec.family
    raise KeyError(f"Unknown reference scenario key: {scenario_key!r}")


def target_nfes_for_role(nfe_role: str) -> Tuple[int, ...]:
    role = str(nfe_role).strip().lower()
    if role == NFE_ROLE_SEEN:
        return REFERENCE_SEEN_NFES
    if role == NFE_ROLE_UNSEEN:
        return REFERENCE_UNSEEN_NFES
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


__all__ = [
    "AVERAGED_REVERSED_SCHEDULE_KEYS",
    "AVERAGED_SCHEDULE_COMPONENTS",
    "REFERENCE_CHECKPOINT_STEPS",
    "TRAIN_TUNING_CONTEXT_SAMPLE_COUNT",
    "EXPERIMENT_LAYOUT_ID",
    "REFERENCE_UNSEEN_TARGET_WEIGHT",
    "REFERENCE_SCENARIO_KEYS",
    "REFERENCE_SEEN_NFES",
    "REFERENCE_SUPERVISION_SCHEDULE_KEYS",
    "REFERENCE_UNSEEN_NFES",
    "CONDITIONAL_GENERATION_SCENARIO_KEYS",
    "LOCKED_TEST_PREVIEW_CONTEXTS",
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
    "STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGET",
    "STUDENT_TRAINING_MODES",
    "ScenarioSpec",
    "scenario_specs",
    "target_nfes_for_role",
    "density_source_key_for_schedule",
    "scenario_family_for_key",
    "schedule_family_for_key",
]
