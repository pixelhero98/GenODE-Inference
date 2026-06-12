from __future__ import annotations

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


def solver_id_map(keys: Iterable[str] = CANONICAL_SOLVER_KEYS) -> Dict[str, int]:
    normalized = normalize_solver_keys(tuple(keys), reject_duplicates=True)
    return {key: idx for idx, key in enumerate(normalized)}


__all__ = [
    "CANONICAL_SOLVER_DISPLAY_NAMES",
    "CANONICAL_SOLVER_KEYS",
    "CANONICAL_SOLVER_RUNTIME_NAMES",
    "SOLVER_ALIASES",
    "expected_realized_nfe",
    "normalize_solver_key",
    "normalize_solver_keys",
    "solver_display_name",
    "solver_effective_order",
    "solver_eval_multiplier",
    "solver_id_map",
    "solver_macro_steps",
    "solver_order_p",
    "solver_runtime_name",
]
