from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from genode.schedule_transfer.diffusion_flow_schedules import (
    BASELINE_SCHEDULE_KEYS,
    TRANSFER_SCHEDULE_KEYS,
    build_schedule_grid,
    load_external_schedule_catalog,
    schedule_display_name,
    schedule_time_alignment,
)

MAIN_NFE_VALUES: Tuple[int, ...] = (4, 8, 12)
APPENDIX_NFE_VALUES: Tuple[int, ...] = (6, 8, 20, 24)
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
    external_catalog = load_external_schedule_catalog()
    return [
        ScheduleSpec(
            key="uniform",
            display_name="Time-uniform",
            family="uniform",
            comparison_role="deterministic_baseline",
            solver_scope="all_fixed_grid_ode",
            implementation_status="implemented",
        ),
        ScheduleSpec(
            key="late_power_3",
            display_name="Late-power-3",
            family="hand_designed",
            comparison_role="deterministic_baseline",
            solver_scope="all_fixed_grid_ode",
            implementation_status="implemented",
        ),
        ScheduleSpec(
            key="flowts_power_sampling",
            display_name="FlowTS power sampling",
            family="flow_matching_schedule_transfer",
            comparison_role="external_flow_matching_schedule",
            solver_scope="all_fixed_grid_ode",
            implementation_status="implemented",
            source_url=external_catalog.get("flowts_power_sampling", {}).get("source_url"),
            paper_url=external_catalog.get("flowts_power_sampling", {}).get("paper_url"),
            external_mapping_status=external_catalog.get("flowts_power_sampling", {}).get("mapping_status"),
        ),
        ScheduleSpec(
            key="ays",
            display_name="AYS",
            family="diffusion_schedule_transfer",
            comparison_role="transferred_optimized_diffusion_schedule",
            solver_scope="all_fixed_grid_ode",
            implementation_status="implemented",
            source_url=external_catalog.get("ays", {}).get("source_url"),
            paper_url=external_catalog.get("ays", {}).get("paper_url"),
            external_mapping_status=external_catalog.get("ays", {}).get("mapping_status"),
        ),
        ScheduleSpec(
            key="gits",
            display_name="GITS",
            family="diffusion_schedule_transfer",
            comparison_role="transferred_optimized_diffusion_schedule",
            solver_scope="all_fixed_grid_ode",
            implementation_status="implemented",
            source_url=external_catalog.get("gits", {}).get("source_url"),
            paper_url=external_catalog.get("gits", {}).get("paper_url"),
            external_mapping_status=external_catalog.get("gits", {}).get("mapping_status"),
        ),
        ScheduleSpec(
            key="ots",
            display_name="OTS",
            family="diffusion_schedule_transfer",
            comparison_role="transferred_optimized_diffusion_schedule",
            solver_scope="all_fixed_grid_ode",
            implementation_status="implemented",
            source_url=external_catalog.get("ots", {}).get("source_url"),
            paper_url=external_catalog.get("ots", {}).get("paper_url"),
            external_mapping_status=external_catalog.get("ots", {}).get("mapping_status"),
        ),
    ]


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
