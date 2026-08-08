from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from numbers import Integral
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


AYS_SD15_TIMESTEPS: Tuple[int, ...] = (999, 850, 736, 645, 545, 455, 343, 233, 124, 24)
AYS_SD15_SIGMAS: Tuple[float, ...] = (14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.0)
SD15_NUM_TRAIN_TIMESTEPS = 1000
SD15_BETA_START = 0.00085
SD15_BETA_END = 0.012
SD15_BETA_SCHEDULE = "scaled_linear"
GITS_CIFAR10_SIGMAS: Tuple[float, ...] = (80.0, 10.9836, 3.8811, 1.5840, 0.5666, 0.1698, 0.0020)
OTS_VP_LINEAR_BETA_0 = 0.1
OTS_VP_LINEAR_BETA_1 = 20.0
OTS_VP_LINEAR_EPS = 1e-3
FLOWTS_POWER = 0.03

DEFAULT_LATE_P_VALUES: Tuple[Decimal, ...] = tuple(Decimal(value) for value in ("1.5", "2", "4", "8"))

REFERENCE_CLOCK_BASE_KEYS: Tuple[str, ...] = (
    "uniform",
    "ays_sd15_native",
    "ays_sd15_log_sigma",
    "gits_cifar10_native",
    "gits_cifar10_log_sigma",
    "ots_vp_linear_native",
    "ots_vp_linear_log_sigma",
    "late_p_1p5",
    "late_p_2",
    "late_p_4",
    "late_p_8",
    "flowts_power_0p03",
)
REFERENCE_CLOCK_REVERSED_KEYS: Tuple[str, ...] = tuple(
    f"{key}_reversed" for key in REFERENCE_CLOCK_BASE_KEYS if key != "uniform"
)
DEFAULT_REFERENCE_CLOCK_KEYS: Tuple[str, ...] = REFERENCE_CLOCK_BASE_KEYS + REFERENCE_CLOCK_REVERSED_KEYS

_LATE_P_KEY_RE = re.compile(r"^late_p_(?P<value>[0-9]+(?:p[0-9]+)?)$")


@dataclass(frozen=True)
class ReferenceClockSpec:
    key: str
    display_name: str
    family: str
    coordinate: str
    realization: str
    application_behavior: str
    source_model: str
    source_solver: str
    source_coordinate: str
    source_repo: str
    source_commit: str
    source_license: str
    source_path: str
    source_nodes: Tuple[float, ...] = ()
    derivation: str = "source_reference"
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["source_nodes"] = list(self.source_nodes)
        return payload


_DIFFUSERS_COMMIT = "50e7158093710f9c1b4ea9ff100137a91c9228f3"
_GITS_COMMIT = "68d5ce427f261962b89ce3b0ee8f6b29f0577328"
_OTS_COMMIT = "95d4ac6b8a3d1d389ab63a197e1b05d8512b6a99"
_FLOWTS_COMMIT = "1ec35fb1d3d89d91a1607a9f949a515347d54c8c"


def _base_specs() -> Dict[str, ReferenceClockSpec]:
    specs = {
        "uniform": ReferenceClockSpec(
            key="uniform",
            display_name="Uniform",
            family="uniform",
            coordinate="native_time",
            realization="analytic_exact",
            application_behavior="exact_genode_clock",
            source_model="GenODE normalized integration time",
            source_solver="solver-independent",
            source_coordinate="t in [0, 1]",
            source_repo="",
            source_commit="",
            source_license="project license",
            source_path="",
            derivation="analytic_formula",
            notes="Equally spaced normalized integration times.",
        ),
        "ays_sd15_native": ReferenceClockSpec(
            key="ays_sd15_native",
            display_name="AYS SD1.5 (native timestep)",
            family="ays_sd15",
            coordinate="native",
            realization="exact_source_nodes_then_deterministic_resampling",
            application_behavior="transferred_reference",
            source_model="Stable Diffusion 1.5",
            source_solver="custom-timestep compatible diffusion solver",
            source_coordinate="DDPM timestep index",
            source_repo="https://github.com/huggingface/diffusers",
            source_commit=_DIFFUSERS_COMMIT,
            source_license="Apache-2.0",
            source_path="src/diffusers/schedulers/scheduling_utils.py::AysSchedules[StableDiffusionTimesteps]",
            source_nodes=tuple(float(value) for value in AYS_SD15_TIMESTEPS),
            notes="The published ten denoiser timesteps are preserved; t=0 is appended only as the integration boundary.",
        ),
        "ays_sd15_log_sigma": ReferenceClockSpec(
            key="ays_sd15_log_sigma",
            display_name="AYS SD1.5 (log-sigma)",
            family="ays_sd15",
            coordinate="log_sigma",
            realization="exact_source_nodes_then_coordinate_transfer",
            application_behavior="transferred_reference",
            source_model="Stable Diffusion 1.5",
            source_solver="custom-sigma compatible diffusion solver",
            source_coordinate="log(sigma/alpha) derived at published SD1.5 nodes",
            source_repo="https://github.com/huggingface/diffusers",
            source_commit=_DIFFUSERS_COMMIT,
            source_license="Apache-2.0",
            source_path=(
                "src/diffusers/schedulers/scheduling_utils.py::AysSchedules[StableDiffusionSigmas]; "
                "src/diffusers/schedulers/scheduling_ddim.py::DDIMScheduler scaled_linear"
            ),
            source_nodes=tuple(float(value) for value in AYS_SD15_SIGMAS),
            derivation="pinned_source_nodes_with_pinned_sd15_scaled_linear_terminal_then_coordinate_transfer",
            notes=(
                "The scheduler-only zero terminal is replaced solely to make log-sigma finite by the SD1.5 t=0 "
                "sigma/alpha derived from the pinned Diffusers scaled-linear realization: 1000 training steps, "
                "beta_start=0.00085, beta_end=0.012."
            ),
        ),
        "gits_cifar10_native": ReferenceClockSpec(
            key="gits_cifar10_native",
            display_name="GITS CIFAR-10 example (native sigma)",
            family="gits_cifar10",
            coordinate="native",
            realization="exact_source_nodes_then_deterministic_resampling",
            application_behavior="transferred_reference",
            source_model="EDM-preconditioned CIFAR-10",
            source_solver="iPNDM, max order 4, six sampling steps",
            source_coordinate="EDM sigma",
            source_repo="https://github.com/zju-pi/diff-sampler",
            source_commit=_GITS_COMMIT,
            source_license="Apache-2.0",
            source_path="gits-main/README.md::pre-specified CIFAR-10 t_steps example",
            source_nodes=GITS_CIFAR10_SIGMAS,
            notes="This is the published CIFAR-10 example, not a GITS schedule optimized for a GenODE backbone.",
        ),
        "gits_cifar10_log_sigma": ReferenceClockSpec(
            key="gits_cifar10_log_sigma",
            display_name="GITS CIFAR-10 example (log-sigma)",
            family="gits_cifar10",
            coordinate="log_sigma",
            realization="exact_source_nodes_then_coordinate_transfer",
            application_behavior="transferred_reference",
            source_model="EDM-preconditioned CIFAR-10",
            source_solver="iPNDM, max order 4, six sampling steps",
            source_coordinate="log(EDM sigma)",
            source_repo="https://github.com/zju-pi/diff-sampler",
            source_commit=_GITS_COMMIT,
            source_license="Apache-2.0",
            source_path="gits-main/README.md::pre-specified CIFAR-10 t_steps example",
            source_nodes=GITS_CIFAR10_SIGMAS,
            notes="A coordinate view of the transferred published CIFAR-10 example.",
        ),
        "ots_vp_linear_native": ReferenceClockSpec(
            key="ots_vp_linear_native",
            display_name="OTS linear VP (native time)",
            family="ots_vp_linear",
            coordinate="native",
            realization="official_objective_reimplementation",
            application_behavior="transferred_reference",
            source_model="continuous linear VP, beta(t)=0.1..20.0",
            source_solver="DPM-Solver/UniPC; uniform-time initialization",
            source_coordinate="VP time t from 1 to 1e-3",
            source_repo="https://github.com/scxue/DM-NonUniform",
            source_commit=_OTS_COMMIT,
            source_license="MIT",
            source_path="step_optim.py::NoiseScheduleVP,StepOptim.get_ts_lambdas",
            notes="Official linear-VP optimizer semantics; applying its nodes to GenODE is a transfer.",
        ),
        "ots_vp_linear_log_sigma": ReferenceClockSpec(
            key="ots_vp_linear_log_sigma",
            display_name="OTS linear VP (log sigma/alpha)",
            family="ots_vp_linear",
            coordinate="log_sigma",
            realization="official_objective_reimplementation",
            application_behavior="transferred_reference",
            source_model="continuous linear VP, beta(t)=0.1..20.0",
            source_solver="DPM-Solver/UniPC; uniform-time initialization",
            source_coordinate="log(sigma/alpha)=-lambda",
            source_repo="https://github.com/scxue/DM-NonUniform",
            source_commit=_OTS_COMMIT,
            source_license="MIT",
            source_path="step_optim.py::NoiseScheduleVP,StepOptim.get_ts_lambdas",
            notes="The same optimized VP nodes viewed in log(sigma/alpha), then normalized to GenODE progress.",
        ),
        "flowts_power_0p03": ReferenceClockSpec(
            key="flowts_power_0p03",
            display_name="FlowTS power 0.03",
            family="flowts_power",
            coordinate="native",
            realization="exact_source_formula",
            application_behavior="transferred_reference",
            source_model="FM-TS time-series flow matching",
            source_solver="one-step Euler infill update",
            source_coordinate="t=(step/num_timesteps)^0.03",
            source_repo="https://github.com/UNITES-Lab/FlowTS",
            source_commit=_FLOWTS_COMMIT,
            source_license="FMTS MIT",
            source_path="FMTS/Models/interpretable_diffusion/FMTS.py::fast_sample_infill; FMTS/run.sh",
            notes="Uses the released hucfg_Kscale=0.03; the explicit endpoint t=1 closes the integration grid.",
        ),
    }
    for power in DEFAULT_LATE_P_VALUES:
        key = canonical_late_p_key(power)
        specs[key] = _late_p_spec(power)
    return specs


def _late_p_spec(power: Decimal) -> ReferenceClockSpec:
    key = canonical_late_p_key(power)
    return ReferenceClockSpec(
        key=key,
        display_name=f"Late-p {format(power, 'f')}",
        family="late_p",
        coordinate="native",
        realization="analytic_exact",
        application_behavior="exact_genode_clock",
        source_model="GenODE normalized integration time",
        source_solver="solver-independent",
        source_coordinate="1-(1-u)^p",
        source_repo="",
        source_commit="",
        source_license="project license",
        source_path="",
        derivation="analytic_formula",
        notes="User augmentation is restricted to finite p in [1.5, 8].",
    )


def validate_late_p_value(value: Decimal | float | int | str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("late-p values must be numeric, not booleans.")
    try:
        power = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid late-p value {value!r}.") from exc
    if not power.is_finite() or power < Decimal("1.5") or power > Decimal("8"):
        raise ValueError(f"late-p must be finite and within [1.5, 8], got {value!r}.")
    return power.normalize()


def canonical_late_p_key(value: Decimal | float | int | str) -> str:
    power = validate_late_p_value(value)
    decimal_text = format(power, "f")
    if "." in decimal_text:
        decimal_text = decimal_text.rstrip("0").rstrip(".")
    return f"late_p_{decimal_text.replace('.', 'p')}"


def late_p_value_from_key(key: str) -> Decimal:
    match = _LATE_P_KEY_RE.fullmatch(str(key).strip().lower())
    if match is None:
        raise KeyError(f"Not a late-p reference clock key: {key!r}.")
    return validate_late_p_value(match.group("value").replace("p", "."))


def parse_extra_late_p_values(values: str | Sequence[Decimal | float | int | str]) -> Tuple[Decimal, ...]:
    raw_values: Sequence[Decimal | float | int | str]
    if isinstance(values, str):
        raw_values = tuple(item.strip() for item in values.split(",") if item.strip())
    else:
        raw_values = values
    return tuple(sorted({validate_late_p_value(value) for value in raw_values}))


def reference_clock_keys(
    extra_late_p_values: str | Sequence[Decimal | float | int | str] = (),
) -> Tuple[str, ...]:
    powers = tuple(sorted(set(DEFAULT_LATE_P_VALUES) | set(parse_extra_late_p_values(extra_late_p_values))))
    base_keys = (
        *REFERENCE_CLOCK_BASE_KEYS[:7],
        *(canonical_late_p_key(power) for power in powers),
        "flowts_power_0p03",
    )
    return tuple(base_keys) + tuple(f"{key}_reversed" for key in base_keys if key != "uniform")


def reference_clock_registry(
    extra_late_p_values: str | Sequence[Decimal | float | int | str] = (),
) -> Mapping[str, ReferenceClockSpec]:
    base_specs = _base_specs()
    for power in parse_extra_late_p_values(extra_late_p_values):
        base_specs.setdefault(canonical_late_p_key(power), _late_p_spec(power))
    registry: Dict[str, ReferenceClockSpec] = {}
    for key in reference_clock_keys(extra_late_p_values):
        if key.endswith("_reversed"):
            base_key = key.removesuffix("_reversed")
            base = base_specs[base_key]
            registry[key] = replace(
                base,
                key=key,
                display_name=f"{base.display_name} reversed",
                derivation="reverse_clock: 1 - reversed(base_grid)",
                notes=f"{base.notes} Reversal is deterministic and preserves endpoints.",
            )
        else:
            registry[key] = base_specs[key]
    return MappingProxyType(registry)


def reference_clock_provenance(key: str) -> Dict[str, Any]:
    normalized = str(key).strip().lower()
    base_key = normalized.removesuffix("_reversed")
    extras: Tuple[Decimal, ...] = ()
    if base_key.startswith("late_p_"):
        extras = (late_p_value_from_key(base_key),)
    registry = reference_clock_registry(extras)
    try:
        return registry[normalized].as_dict()
    except KeyError as exc:
        raise KeyError(f"Unknown reference clock {key!r}.") from exc


def _validate_n_steps(n_steps: int) -> int:
    if isinstance(n_steps, bool) or not isinstance(n_steps, Integral):
        raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}.")
    n_steps = int(n_steps)
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}.")
    return n_steps


def _normalize_descending(values: Sequence[float]) -> Tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not bool(np.all(np.isfinite(array))):
        raise ValueError("Reference nodes must be a finite one-dimensional sequence with at least two values.")
    if not bool(np.all(np.diff(array) < 0.0)):
        raise ValueError("Reference nodes must be strictly descending.")
    progression = (array[0] - array) / (array[0] - array[-1])
    progression[0], progression[-1] = 0.0, 1.0
    return tuple(float(value) for value in progression)


def _resample(progression: Sequence[float], n_steps: int) -> Tuple[float, ...]:
    reference = np.asarray(progression, dtype=np.float64)
    src = np.linspace(0.0, 1.0, reference.size, dtype=np.float64)
    dst = np.linspace(0.0, 1.0, int(n_steps) + 1, dtype=np.float64)
    return _finalize(np.interp(dst, src, reference))


def _finalize(values: Sequence[float]) -> Tuple[float, ...]:
    grid = np.asarray(values, dtype=np.float64).copy()
    if grid.ndim != 1 or grid.size < 2 or not bool(np.all(np.isfinite(grid))):
        raise ValueError("Reference clock grid must be finite and one-dimensional.")
    grid[0], grid[-1] = 0.0, 1.0
    if not bool(np.all(np.diff(grid) > 0.0)):
        raise ValueError("Reference clock grid must be strictly increasing.")
    return tuple(float(value) for value in grid)


def reverse_reference_clock_grid(grid: Sequence[float]) -> Tuple[float, ...]:
    return _finalize([1.0 - float(value) for value in reversed(tuple(grid))])


@lru_cache(maxsize=1)
def _sd15_sigma_ratio_t0() -> float:
    if SD15_BETA_SCHEDULE != "scaled_linear":
        raise RuntimeError(f"Unsupported pinned SD1.5 beta schedule {SD15_BETA_SCHEDULE!r}.")
    betas = (
        np.linspace(
            math.sqrt(SD15_BETA_START),
            math.sqrt(SD15_BETA_END),
            SD15_NUM_TRAIN_TIMESTEPS,
            dtype=np.float64,
        )
        ** 2
    )
    alpha_bar_t0 = float(1.0 - betas[0])
    return math.sqrt((1.0 - alpha_bar_t0) / alpha_bar_t0)


def _ays_progression(coordinate: str) -> Tuple[float, ...]:
    if coordinate == "native":
        return _normalize_descending((*AYS_SD15_TIMESTEPS, 0.0))
    positive_sigmas = (*AYS_SD15_SIGMAS[:-1], _sd15_sigma_ratio_t0())
    return _normalize_descending(tuple(math.log(value) for value in positive_sigmas))


def _gits_progression(coordinate: str) -> Tuple[float, ...]:
    values = GITS_CIFAR10_SIGMAS if coordinate == "native" else tuple(math.log(value) for value in GITS_CIFAR10_SIGMAS)
    return _normalize_descending(values)


def _vp_alpha(t: np.ndarray) -> np.ndarray:
    log_alpha = -0.25 * t * t * (OTS_VP_LINEAR_BETA_1 - OTS_VP_LINEAR_BETA_0) - 0.5 * t * OTS_VP_LINEAR_BETA_0
    return np.exp(log_alpha)


def _vp_lambda(t: Sequence[float] | np.ndarray) -> np.ndarray:
    times = np.asarray(t, dtype=np.float64)
    alpha = _vp_alpha(times)
    sigma = np.sqrt(1.0 - alpha * alpha)
    return np.log(alpha / sigma)


def _vp_inverse_lambda(values: Sequence[float] | np.ndarray) -> np.ndarray:
    lambdas = np.asarray(values, dtype=np.float64)
    beta_delta = OTS_VP_LINEAR_BETA_1 - OTS_VP_LINEAR_BETA_0
    temporary = 2.0 * beta_delta * np.logaddexp(-2.0 * lambdas, 0.0)
    return temporary / ((np.sqrt(OTS_VP_LINEAR_BETA_0**2 + temporary) + OTS_VP_LINEAR_BETA_0) * beta_delta)


def _ots_objective(lambda_vec: np.ndarray) -> float:
    lambda_t = float(_vp_lambda([1.0])[0])
    lambda_eps = float(_vp_lambda([OTS_VP_LINEAR_EPS])[0])
    nodes = np.concatenate(([lambda_t], np.asarray(lambda_vec, dtype=np.float64), [lambda_eps]))
    h = np.diff(nodes)
    exp_lambda = np.exp(nodes)
    exp_minus_two_lambda = np.exp(-2.0 * nodes)
    alpha = 1.0 / np.sqrt(1.0 + exp_minus_two_lambda)
    sigma = 1.0 / np.sqrt(1.0 + np.exp(2.0 * nodes))
    data_error = sigma**2 / alpha
    coefficients = np.zeros(len(nodes) - 1, dtype=np.float64)
    result = 0.0

    def h0(value: float) -> float:
        return math.exp(value) - 1.0

    def h1(value: float) -> float:
        return math.exp(value) * value - h0(value)

    def h2(value: float) -> float:
        return math.exp(value) * value * value - 2.0 * h1(value)

    n_intervals = len(nodes) - 1
    for step in range(n_intervals):
        if step in (0, n_intervals - 1):
            result += abs((exp_lambda[step + 1] - exp_lambda[step]) * data_error[step])
        elif step in (1, n_intervals - 2):
            index = step - 1
            j0 = -exp_lambda[index + 1] * h1(h[index + 1]) / h[index]
            j1 = exp_lambda[index + 1] * (h1(h[index + 1]) + h[index] * h0(h[index + 1])) / h[index]
            if step >= 3:
                coefficients[index] += data_error[index] * j0
                coefficients[index + 1] += data_error[index + 1] * j1
            else:
                result += math.hypot(data_error[index] * j0, data_error[index + 1] * j1)
        else:
            index = step - 2
            j0 = exp_lambda[index + 2] * (h2(h[index + 2]) + h[index + 1] * h1(h[index + 2])) / (h[index] * (h[index] + h[index + 1]))
            j1 = -exp_lambda[index + 2] * (h2(h[index + 2]) + (h[index] + h[index + 1]) * h1(h[index + 2])) / (h[index] * h[index + 1])
            j2 = exp_lambda[index + 2] * (
                h2(h[index + 2])
                + (2.0 * h[index + 1] + h[index]) * h1(h[index + 2])
                + h[index + 1] * (h[index] + h[index + 1]) * h0(h[index + 2])
            ) / (h[index + 1] * (h[index] + h[index + 1]))
            if step >= 3:
                coefficients[index] += data_error[index] * j0
                coefficients[index + 1] += data_error[index + 1] * j1
                coefficients[index + 2] += data_error[index + 2] * j2
            else:
                result += math.sqrt(
                    (data_error[index] * j0) ** 2
                    + (data_error[index + 1] * j1) ** 2
                    + (data_error[index + 2] * j2) ** 2
                )
    return float(result + np.sum(np.abs(coefficients)))


@lru_cache(maxsize=None)
def ots_vp_linear_source_nodes(n_steps: int) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    n_steps = _validate_n_steps(n_steps)
    lambda_t = float(_vp_lambda([1.0])[0])
    lambda_eps = float(_vp_lambda([OTS_VP_LINEAR_EPS])[0])
    if n_steps == 1:
        lambdas = np.asarray([lambda_t, lambda_eps], dtype=np.float64)
        return (1.0, OTS_VP_LINEAR_EPS), tuple(float(value) for value in lambdas)

    try:
        from scipy.optimize import LinearConstraint, minimize
    except ImportError as exc:
        raise RuntimeError("OTS reference clocks require scipy.optimize.") from exc

    constraint_matrix = np.zeros((n_steps, n_steps - 1), dtype=np.float64)
    for index in range(n_steps - 1):
        constraint_matrix[index, index] = 1.0
        constraint_matrix[index + 1, index] = -1.0
    lower = np.zeros(n_steps, dtype=np.float64)
    lower[0], lower[-1] = lambda_t, -lambda_eps
    constraint = LinearConstraint(constraint_matrix, lower, np.full(n_steps, np.inf, dtype=np.float64))
    initial_times = np.linspace(1.0, OTS_VP_LINEAR_EPS, n_steps + 1, dtype=np.float64)
    initial_lambdas = _vp_lambda(initial_times)[1:-1]
    result = minimize(
        _ots_objective,
        initial_lambdas,
        method="trust-constr",
        constraints=[constraint],
        options={"verbose": 0},
    )
    if not bool(result.success) or not bool(np.all(np.isfinite(result.x))):
        raise RuntimeError(f"OTS optimizer failed: {result.message}")
    lambdas = np.concatenate(([lambda_t], np.asarray(result.x, dtype=np.float64), [lambda_eps]))
    times = _vp_inverse_lambda(lambdas)
    times[0], times[-1] = 1.0, OTS_VP_LINEAR_EPS
    return tuple(float(value) for value in times), tuple(float(value) for value in lambdas)


def build_reference_clock_grid(key: str, n_steps: int) -> Tuple[float, ...]:
    normalized = str(key).strip().lower()
    n_steps = _validate_n_steps(n_steps)
    base_key = normalized.removesuffix("_reversed")
    try:
        extra_late_p_values: Tuple[Decimal, ...] = (
            (late_p_value_from_key(base_key),) if base_key.startswith("late_p_") else ()
        )
        registry = reference_clock_registry(extra_late_p_values)
    except (KeyError, ValueError) as exc:
        raise KeyError(f"Unknown reference clock {key!r}.") from exc
    if normalized not in registry:
        raise KeyError(f"Unknown reference clock {key!r}.")
    if normalized.endswith("_reversed"):
        return reverse_reference_clock_grid(
            build_reference_clock_grid(base_key, n_steps)
        )
    if normalized == "uniform":
        return _finalize(np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float64))
    if normalized.startswith("late_p_"):
        power = float(late_p_value_from_key(normalized))
        unit = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float64)
        return _finalize(1.0 - (1.0 - unit) ** power)
    if normalized == "flowts_power_0p03":
        unit = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float64)
        return _finalize(unit**FLOWTS_POWER)
    if normalized in ("ays_sd15_native", "ays_sd15_log_sigma"):
        coordinate = "log_sigma" if normalized.endswith("log_sigma") else "native"
        return _resample(_ays_progression(coordinate), n_steps)
    if normalized in ("gits_cifar10_native", "gits_cifar10_log_sigma"):
        coordinate = "log_sigma" if normalized.endswith("log_sigma") else "native"
        return _resample(_gits_progression(coordinate), n_steps)
    if normalized in ("ots_vp_linear_native", "ots_vp_linear_log_sigma"):
        times, lambdas = ots_vp_linear_source_nodes(n_steps)
        values = times if normalized.endswith("native") else tuple(-value for value in lambdas)
        return _finalize(_normalize_descending(values))
    raise KeyError(f"Unknown reference clock {key!r}.")


__all__ = [
    "AYS_SD15_SIGMAS",
    "AYS_SD15_TIMESTEPS",
    "DEFAULT_LATE_P_VALUES",
    "DEFAULT_REFERENCE_CLOCK_KEYS",
    "FLOWTS_POWER",
    "GITS_CIFAR10_SIGMAS",
    "OTS_VP_LINEAR_BETA_0",
    "OTS_VP_LINEAR_BETA_1",
    "OTS_VP_LINEAR_EPS",
    "REFERENCE_CLOCK_BASE_KEYS",
    "REFERENCE_CLOCK_REVERSED_KEYS",
    "SD15_BETA_END",
    "SD15_BETA_SCHEDULE",
    "SD15_BETA_START",
    "SD15_NUM_TRAIN_TIMESTEPS",
    "ReferenceClockSpec",
    "build_reference_clock_grid",
    "canonical_late_p_key",
    "late_p_value_from_key",
    "ots_vp_linear_source_nodes",
    "parse_extra_late_p_values",
    "reference_clock_keys",
    "reference_clock_provenance",
    "reference_clock_registry",
    "reverse_reference_clock_grid",
    "validate_late_p_value",
]
