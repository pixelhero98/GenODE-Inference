from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

CANONICAL_SOLVER_KEYS: Tuple[str, ...] = ("euler", "dpmpp2m", "heun", "midpoint_rk2")
CANONICAL_SOLVER_DISPLAY_NAMES: Dict[str, str] = {
    "euler": "Euler",
    "dpmpp2m": "DPM++2M",
    "heun": "Heun / RK2",
    "midpoint_rk2": "Midpoint RK2",
}
CANONICAL_SOLVER_RUNTIME_NAMES: Dict[str, str] = {
    "euler": "euler",
    "dpmpp2m": "dpmpp2m",
    "heun": "heun",
    "midpoint_rk2": "midpoint_rk2",
}
SOLVER_ALIASES: Dict[str, str] = {
    "euler": "euler",
    "dpmpp2m": "dpmpp2m",
    "dpmpp_2m": "dpmpp2m",
    "dpm++": "dpmpp2m",
    "dpm++2m": "dpmpp2m",
    "dpm++_2m": "dpmpp2m",
    "heun": "heun",
    "rk2": "heun",
    "heun_rk2": "heun",
    "heun/rk2": "heun",
    "midpoint_rk2": "midpoint_rk2",
    "midpoint-rk2": "midpoint_rk2",
    "rk2_midpoint": "midpoint_rk2",
    "midpoint rk2": "midpoint_rk2",
}


def normalize_solver_key(value: str) -> str:
    key = str(value).strip().lower()
    if key not in SOLVER_ALIASES:
        raise ValueError(f"Unknown solver_key={value!r}; expected one of {CANONICAL_SOLVER_KEYS}.")
    return SOLVER_ALIASES[key]


def normalize_solver_keys(values: Sequence[str] | str, *, reject_duplicates: bool = True) -> Tuple[str, ...]:
    if isinstance(values, str):
        raw = [part.strip() for part in values.split(",") if part.strip()]
    else:
        raw = [str(part).strip() for part in values if str(part).strip()]
    if not raw:
        raise ValueError("At least one solver key is required.")
    normalized = tuple(normalize_solver_key(part) for part in raw)
    if reject_duplicates:
        duplicates = sorted({key for key in normalized if normalized.count(key) > 1})
        if duplicates:
            raise ValueError(f"Duplicate solver keys after alias normalization: {duplicates}.")
    return normalized


def solver_runtime_name(solver_key: str) -> str:
    key = normalize_solver_key(solver_key)
    return CANONICAL_SOLVER_RUNTIME_NAMES[key]


def solver_display_name(solver_key: str) -> str:
    key = normalize_solver_key(solver_key)
    return CANONICAL_SOLVER_DISPLAY_NAMES[key]


def solver_eval_multiplier(solver_key: str) -> int:
    key = normalize_solver_key(solver_key)
    return 2 if key in {"heun", "midpoint_rk2"} else 1


def solver_macro_steps(solver_key: str, target_nfe: int) -> int:
    key = normalize_solver_key(solver_key)
    target = int(target_nfe)
    if target <= 0:
        raise ValueError(f"target_nfe must be positive, got {target_nfe!r}.")
    if key in {"heun", "midpoint_rk2"}:
        if target % 2 != 0:
            raise ValueError(f"{key} requires an even target NFE, got {target}.")
        return target // 2
    return target


def solver_effective_order(solver_key: str) -> int:
    return 1 if normalize_solver_key(solver_key) == "euler" else 2


def solver_order_p(solver_key: str) -> float:
    return float(solver_effective_order(solver_key))


def expected_realized_nfe(solver_key: str, target_nfe: int) -> int:
    key = normalize_solver_key(solver_key)
    return int(solver_macro_steps(key, int(target_nfe)) * solver_eval_multiplier(key))


@dataclass(frozen=True)
class SolverNFEFields:
    solver_key: str
    target_nfe: int
    macro_steps: int
    runtime_nfe: int
    realized_nfe: int


def _optional_positive_int(value: object, *, field: str, source: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} has non-integer {field}={value!r}.") from exc
    if parsed <= 0:
        raise ValueError(f"{source} requires positive {field}; got {parsed}.")
    return parsed


def normalize_solver_nfe_fields(
    solver_key: str,
    target_nfe: int,
    *,
    macro_steps: object = None,
    runtime_nfe: object = None,
    realized_nfe: object = None,
    source: str = "row",
) -> SolverNFEFields:
    """Return canonical macro/runtime/realized NFE fields for one solver target.

    ``runtime_nfe`` is the legacy runtime argument and therefore equals macro
    steps.  ``realized_nfe`` records actual model evaluations.
    """

    key = normalize_solver_key(str(solver_key))
    target = int(target_nfe)
    expected_macro = solver_macro_steps(key, target)
    expected_realized = expected_realized_nfe(key, target)
    parsed_macro = _optional_positive_int(macro_steps, field="macro_steps", source=source)
    parsed_runtime = _optional_positive_int(runtime_nfe, field="runtime_nfe", source=source)
    parsed_realized = _optional_positive_int(realized_nfe, field="realized_nfe", source=source)
    if parsed_macro is not None and parsed_macro != expected_macro:
        raise ValueError(
            f"{source} has macro_steps={parsed_macro} for {key}/{target}; expected {expected_macro}."
        )
    if parsed_runtime is not None and parsed_runtime != expected_macro:
        raise ValueError(
            f"{source} has runtime_nfe={parsed_runtime} for {key}/{target}; "
            f"runtime_nfe is macro-step count and must equal {expected_macro}. "
            f"Use realized_nfe={expected_realized} for actual function evaluations."
        )
    if parsed_realized is not None and parsed_realized != expected_realized:
        raise ValueError(
            f"{source} has realized_nfe={parsed_realized} for {key}/{target}; expected {expected_realized}."
        )
    return SolverNFEFields(
        solver_key=key,
        target_nfe=target,
        macro_steps=int(expected_macro),
        runtime_nfe=int(expected_macro),
        realized_nfe=int(expected_realized),
    )


def solver_id_map(keys: Iterable[str] = CANONICAL_SOLVER_KEYS) -> Dict[str, int]:
    normalized = normalize_solver_keys(tuple(keys), reject_duplicates=True)
    return {key: idx for idx, key in enumerate(normalized)}


__all__ = [
    "CANONICAL_SOLVER_DISPLAY_NAMES",
    "CANONICAL_SOLVER_KEYS",
    "CANONICAL_SOLVER_RUNTIME_NAMES",
    "SOLVER_ALIASES",
    "SolverNFEFields",
    "expected_realized_nfe",
    "normalize_solver_key",
    "normalize_solver_keys",
    "normalize_solver_nfe_fields",
    "solver_display_name",
    "solver_effective_order",
    "solver_eval_multiplier",
    "solver_id_map",
    "solver_macro_steps",
    "solver_order_p",
    "solver_runtime_name",
]
