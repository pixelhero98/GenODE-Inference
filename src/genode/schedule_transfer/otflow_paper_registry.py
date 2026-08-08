from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from genode.canonical_experiment_layout import (
    CANONICAL_SEEN_NFES,
    CANONICAL_UNSEEN_NFES,
    PHYSICAL_SCHEDULE_KEYS,
    REVERSED_SCHEDULE_KEYS,
)
from genode.schedule_transfer.diffusion_flow_schedules import (
    BASELINE_SCHEDULE_KEYS,
    TRANSFER_SCHEDULE_KEYS,
    build_schedule_grid,
    load_external_schedule_catalog,
    schedule_display_name,
    schedule_time_alignment,
)
from genode.schedule_transfer.reference_clocks import reference_clock_provenance

MAIN_NFE_VALUES: Tuple[int, ...] = CANONICAL_SEEN_NFES
APPENDIX_NFE_VALUES: Tuple[int, ...] = CANONICAL_UNSEEN_NFES
METHOD_KEY = "diffusion_flow_time_reparameterization"
PAPER_MAIN_SIGNAL_FAMILY = "native_info_growth"


@dataclass(frozen=True)
class ScheduleSpec:
    key: str
    display_name: str
    family: str
    comparison_role: str
    solver_scope: str
    implementation_status: str
    source_url: Optional[str] = None
    paper_url: Optional[str] = None
    requires_signal: bool = False
    external_mapping_status: Optional[str] = None


@dataclass(frozen=True)
class SolverSpec:
    key: str
    display_name: str
    order: int
    family: str
    implementation_status: str
    main_matrix_scope: str
    otflow_runtime_name: Optional[str] = None


def paper_schedule_specs() -> List[ScheduleSpec]:
    specs: List[ScheduleSpec] = []
    for key in BASELINE_SCHEDULE_KEYS:
        provenance = reference_clock_provenance(key)
        transferred = provenance["application_behavior"] == "transferred_reference"
        specs.append(
            ScheduleSpec(
                key=key,
                display_name=str(provenance["display_name"]),
                family=str(provenance["family"]),
                comparison_role="transferred_reference_clock" if transferred else "deterministic_baseline",
                solver_scope="all_fixed_grid_ode",
                implementation_status="implemented",
                source_url=str(provenance["source_repo"]) or None,
                external_mapping_status=str(provenance["application_behavior"]),
            )
        )
    return specs


def paper_solver_specs() -> List[SolverSpec]:
    return [
        SolverSpec("euler", "Euler", 1, "deterministic_ode", "available", "all_schedules", "euler"),
        SolverSpec("heun", "Heun / RK2", 2, "deterministic_ode", "available", "all_schedules", "heun"),
        SolverSpec("midpoint_rk2", "Midpoint RK2", 2, "deterministic_ode", "available", "all_schedules", "midpoint_rk2"),
        SolverSpec("dpmpp2m", "DPM++2M", 2, "multistep_ode", "available", "all_schedules", "dpmpp2m"),
    ]


def paper_registry_snapshot() -> Dict[str, Any]:
    return {
        "method_key": METHOD_KEY,
        "paper_method": METHOD_KEY,
        "main_nfe_values": list(MAIN_NFE_VALUES),
        "appendix_nfe_values": list(APPENDIX_NFE_VALUES),
        "paper_main_signal_family": PAPER_MAIN_SIGNAL_FAMILY,
        "baseline_schedule_keys": list(BASELINE_SCHEDULE_KEYS),
        "physical_schedule_keys": list(PHYSICAL_SCHEDULE_KEYS),
        "canonical_reversed_schedule_keys": list(REVERSED_SCHEDULE_KEYS),
        "transfer_schedule_keys": list(TRANSFER_SCHEDULE_KEYS),
        "schedules": [asdict(spec) for spec in paper_schedule_specs()],
        "solvers": [asdict(spec) for spec in paper_solver_specs()],
    }


__all__ = [
    "APPENDIX_NFE_VALUES",
    "BASELINE_SCHEDULE_KEYS",
    "MAIN_NFE_VALUES",
    "METHOD_KEY",
    "PAPER_MAIN_SIGNAL_FAMILY",
    "ScheduleSpec",
    "SolverSpec",
    "TRANSFER_SCHEDULE_KEYS",
    "build_schedule_grid",
    "load_external_schedule_catalog",
    "paper_registry_snapshot",
    "paper_schedule_specs",
    "paper_solver_specs",
    "schedule_display_name",
    "schedule_time_alignment",
]
