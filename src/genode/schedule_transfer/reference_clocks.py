from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, InvalidOperation
from functools import cache, lru_cache
from numbers import Integral
from types import MappingProxyType
from typing import Any

import numpy as np

from genode.artifacts.identity import semantic_sha256

AYS_SD15_TIMESTEPS: tuple[int, ...] = (999, 850, 736, 645, 545, 455, 343, 233, 124, 24)
AYS_SD15_SIGMAS: tuple[float, ...] = (14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.0)
SD15_NUM_TRAIN_TIMESTEPS = 1000
SD15_BETA_START = 0.00085
SD15_BETA_END = 0.012
SD15_BETA_SCHEDULE = "scaled_linear"
GITS_CIFAR10_SIGMAS: tuple[float, ...] = (80.0, 10.9836, 3.8811, 1.5840, 0.5666, 0.1698, 0.0020)
OTS_VP_LINEAR_BETA_0 = 0.1
OTS_VP_LINEAR_BETA_1 = 20.0
OTS_VP_LINEAR_EPS = 1e-3
FLOWTS_POWER = 0.03

DEFAULT_LATE_P_VALUES: tuple[Decimal, ...] = tuple(Decimal(value) for value in ("1.5", "2", "4", "8"))

REFERENCE_CLOCK_BASE_KEYS: tuple[str, ...] = (
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
REFERENCE_CLOCK_REVERSED_KEYS: tuple[str, ...] = tuple(
    f"{key}_reversed" for key in REFERENCE_CLOCK_BASE_KEYS if key != "uniform"
)
DEFAULT_REFERENCE_CLOCK_KEYS: tuple[str, ...] = REFERENCE_CLOCK_BASE_KEYS + REFERENCE_CLOCK_REVERSED_KEYS

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
    source_nodes: tuple[float, ...] = ()
    derivation: str = "source_reference"
    realization_sha256: str = ""
    realization_environment: str = ""
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_nodes"] = list(self.source_nodes)
        return payload


_DIFFUSERS_COMMIT = "50e7158093710f9c1b4ea9ff100137a91c9228f3"
_GITS_COMMIT = "68d5ce427f261962b89ce3b0ee8f6b29f0577328"
_OTS_COMMIT = "95d4ac6b8a3d1d389ab63a197e1b05d8512b6a99"
_FLOWTS_COMMIT = "1ec35fb1d3d89d91a1607a9f949a515347d54c8c"

OTS_VP_LINEAR_SUPPORTED_STEP_COUNTS: tuple[int, ...] = (
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    10,
    12,
    14,
    16,
    20,
)
OTS_VP_LINEAR_OFFICIAL_TIMES: Mapping[int, tuple[float, ...]] = MappingProxyType(
    {
        2: (0.9999999999999998, 0.2968592779377656, 0.001000000000000059),
        3: (
            0.9999999999999998,
            0.4871723341685362,
            0.2114636736803679,
            0.001000000000000059,
        ),
        4: (
            0.9999999999999998,
            0.6291000431235867,
            0.41664592556647934,
            0.12023062390135601,
            0.001000000000000059,
        ),
        5: (
            0.9999999999999998,
            0.658190396737669,
            0.497844866116979,
            0.3094591651941131,
            0.10136362626069051,
            0.001000000000000059,
        ),
        6: (
            0.9999999999999998,
            0.7106528480563642,
            0.5351537705859608,
            0.42062979153539964,
            0.24027053475200535,
            0.083550784166524,
            0.001000000000000059,
        ),
        7: (
            0.9999999999999998,
            0.7688483584352114,
            0.625333564004778,
            0.5319211467485778,
            0.3404677956461694,
            0.1726471614387146,
            0.06195925711813602,
            0.0010000000000000588,
        ),
        8: (
            0.9999999999999998,
            0.8409368051544779,
            0.7423683355204655,
            0.6743971648516056,
            0.5153113958646132,
            0.33637703646275485,
            0.1728957995830331,
            0.06183390009283608,
            0.001000000000000059,
        ),
        10: (
            0.9999999999999998,
            0.8556787089630243,
            0.7822074406464122,
            0.7405486706278098,
            0.6457873420868624,
            0.5349625034891581,
            0.3659867076077955,
            0.205215945327609,
            0.10898664233449319,
            0.04560919381622387,
            0.001000000000000059,
        ),
        12: (
            0.9999999999999998,
            0.8669631104044307,
            0.7956871870570249,
            0.7534649790129124,
            0.6845395703024086,
            0.6015461560499665,
            0.5099149314932314,
            0.4333900417589155,
            0.297456614963729,
            0.17680117176673427,
            0.10098538898731046,
            0.04529088192567791,
            0.001000000000000059,
        ),
        14: (
            0.9999999999999998,
            0.8987724040731033,
            0.8364261020842262,
            0.7994228033533177,
            0.7463752263696931,
            0.6769521400833034,
            0.5961742407038784,
            0.5122766250948144,
            0.4507101495515668,
            0.3715270483518498,
            0.2588901299081079,
            0.15554278377281855,
            0.09059770811686044,
            0.04114644808221137,
            0.001000000000000059,
        ),
        16: (
            0.9999999999999998,
            0.9067654258228279,
            0.853585230653121,
            0.8264716021289307,
            0.7804037993525473,
            0.7232164406053437,
            0.6567217628862884,
            0.5916140748878191,
            0.52394325275001,
            0.4493671734731429,
            0.3873187285305019,
            0.3230747711843748,
            0.2216022203134519,
            0.13307841235070522,
            0.07642719987716971,
            0.032204350339732964,
            0.001000000000000059,
        ),
        20: (
            0.9999999999999998,
            0.9323888585388722,
            0.8972361554518902,
            0.8778422122292521,
            0.8320133359968838,
            0.7774415281314796,
            0.7123305228681149,
            0.6525918848451091,
            0.5975954607761669,
            0.5447026456645336,
            0.4964786477430131,
            0.4454152650991521,
            0.40282525338778075,
            0.3649830333083829,
            0.31634782784740456,
            0.24496714513737797,
            0.160020787276962,
            0.09215753735962925,
            0.052969757199088315,
            0.023610451007962864,
            0.001000000000000059,
        ),
    }
)
OTS_VP_LINEAR_OFFICIAL_LAMBDAS: Mapping[int, tuple[float, ...]] = MappingProxyType(
    {
        2: (-5.024978406659204, -0.19457526997875002, 4.557714932729866),
        3: (
            -5.024978406659204,
            -1.1580665618456478,
            0.26066235803787174,
            4.557714932729866,
        ),
        4: (
            -5.024978406659204,
            -1.9911597979507625,
            -0.7909855525546063,
            0.8899472408394126,
            4.557714932729866,
        ),
        5: (
            -5.024978406659204,
            -2.1818264330714485,
            -1.2158225647349772,
            -0.25779321904113806,
            1.0646307527024288,
            4.557714932729866,
        ),
        6: (
            -5.024978406659204,
            -2.5449745839537155,
            -1.4233381356651889,
            -0.8111628079555614,
            0.09966575552969238,
            1.2571414781755568,
            4.557714932729866,
        ),
        7: (
            -5.024978406659204,
            -2.9780097879125127,
            -1.9670130505029986,
            -1.4049914624110396,
            -0.4117932430386521,
            0.49891679292980207,
            1.5461919793144823,
            4.557714932729866,
        ),
        8: (
            -5.024978406659204,
            -3.559836671871322,
            -2.7769619985702403,
            -2.2913192882100146,
            -1.3118436352295364,
            -0.39155336923669665,
            0.49728580216192664,
            1.5481178350437772,
            4.557714932729866,
        ),
        10: (
            -5.024978406659204,
            -3.685094773690985,
            -3.082005767547916,
            -2.7633933634928165,
            -2.099622077975055,
            -1.4222505817275104,
            -0.5379871253379569,
            0.29709330387596766,
            0.9910210918535388,
            1.8329604372549202,
            4.557714932729866,
        ),
        12: (
            -5.024978406659204,
            -3.7824235925237986,
            -3.1886978547386606,
            -2.8603919938129097,
            -2.3610557169796387,
            -1.8172935025108519,
            -1.2819716704101976,
            -0.8761693403309915,
            -0.19758468458697115,
            0.4718743586431552,
            1.0684038847883346,
            1.839410164156582,
            4.557714932729866,
        ),
        14: (
            -5.024978406659204,
            -4.063555307500059,
            -3.5219380324380314,
            -3.2185784764907224,
            -2.806951667871525,
            -2.308797973199281,
            -1.7841346456871083,
            -1.2950217144679097,
            -0.9654445000080086,
            -0.5654283150652567,
            0.000368446775468576,
            0.6153043776192275,
            1.1770864191306973,
            1.9273134413549573,
            4.557714932729866,
        ),
        16: (
            -5.024978406659204,
            -4.1357725415724165,
            -3.66717654128337,
            -3.439008916133693,
            -3.0678639825252065,
            -2.6357334079666415,
            -2.1720211868671977,
            -1.7561711943584413,
            -1.36002024733613,
            -0.9584756481852844,
            -0.6438714436314388,
            -0.3256135245978011,
            0.2027892290544528,
            0.7834994385915632,
            1.3443272764006873,
            2.1476728346444087,
            4.557714932729866,
        ),
        20: (
            -5.024978406659204,
            -4.371550863233067,
            -4.049747759956291,
            -3.877447411007579,
            -3.485055915035512,
            -3.044706253964199,
            -2.557008080670721,
            -2.144551534021468,
            -1.7928843193840711,
            -1.4779613543473638,
            -1.208391410064323,
            -0.9380156958671095,
            -0.7213727975558561,
            -0.5330191064957805,
            -0.2921586296314561,
            0.0743211684185572,
            0.5839852619825979,
            1.160102038883269,
            1.6940556594365184,
            2.4179805213825576,
            4.557714932729866,
        ),
    }
)
OTS_VP_LINEAR_REALIZATION_ENVIRONMENT = (
    "Python 3.13.5; NumPy 2.5.1; SciPy 1.18.0; Torch 2.13.0+cpu; torch default dtype float32"
)
OTS_VP_LINEAR_TABLE_SHA256 = semantic_sha256(
    {
        "source_commit": _OTS_COMMIT,
        "source_path": "step_optim.py::StepOptim.get_ts_lambdas(initType='unif_t')",
        "environment": OTS_VP_LINEAR_REALIZATION_ENVIRONMENT,
        "tables_by_source_step_count": [
            {
                "source_step_count": step_count,
                "macro_steps": step_count,
                "times": list(OTS_VP_LINEAR_OFFICIAL_TIMES[step_count]),
                "lambdas": list(OTS_VP_LINEAR_OFFICIAL_LAMBDAS[step_count]),
            }
            for step_count in OTS_VP_LINEAR_SUPPORTED_STEP_COUNTS
        ],
    },
    namespace="ots-vp-linear-official-tables-v2",
)


def _base_specs() -> dict[str, ReferenceClockSpec]:
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
            realization="pinned_official_float32_initialized_tables",
            application_behavior="transferred_reference",
            source_model="continuous linear VP, beta(t)=0.1..20.0",
            source_solver="DPM-Solver/UniPC; uniform-time initialization",
            source_coordinate="VP time t from 1 to 1e-3",
            source_repo="https://github.com/scxue/DM-NonUniform",
            source_commit=_OTS_COMMIT,
            source_license="MIT",
            source_path="step_optim.py::NoiseScheduleVP,StepOptim.get_ts_lambdas",
            derivation="pinned_upstream_runtime_outputs",
            realization_sha256=OTS_VP_LINEAR_TABLE_SHA256,
            realization_environment=OTS_VP_LINEAR_REALIZATION_ENVIRONMENT,
            notes=(
                "Official unif_t outputs are pinned for source step counts "
                "2,3,4,5,6,7,8,10,12,14,16,20, which are used as GenODE macro steps; "
                "applying their nodes to GenODE is a transfer."
            ),
        ),
        "ots_vp_linear_log_sigma": ReferenceClockSpec(
            key="ots_vp_linear_log_sigma",
            display_name="OTS linear VP (log sigma/alpha)",
            family="ots_vp_linear",
            coordinate="log_sigma",
            realization="pinned_official_float32_initialized_tables",
            application_behavior="transferred_reference",
            source_model="continuous linear VP, beta(t)=0.1..20.0",
            source_solver="DPM-Solver/UniPC; uniform-time initialization",
            source_coordinate="log(sigma/alpha)=-lambda",
            source_repo="https://github.com/scxue/DM-NonUniform",
            source_commit=_OTS_COMMIT,
            source_license="MIT",
            source_path="step_optim.py::NoiseScheduleVP,StepOptim.get_ts_lambdas",
            derivation="pinned_upstream_runtime_outputs_then_coordinate_view",
            realization_sha256=OTS_VP_LINEAR_TABLE_SHA256,
            realization_environment=OTS_VP_LINEAR_REALIZATION_ENVIRONMENT,
            notes=(
                "The pinned official lambda_res nodes viewed in log(sigma/alpha), then normalized to GenODE progress."
            ),
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


def parse_extra_late_p_values(values: str | Sequence[Decimal | float | int | str]) -> tuple[Decimal, ...]:
    raw_values: Sequence[Decimal | float | int | str]
    if isinstance(values, str):
        raw_values = tuple(item.strip() for item in values.split(",") if item.strip())
    else:
        raw_values = values
    return tuple(sorted({validate_late_p_value(value) for value in raw_values}))


def reference_clock_keys(
    extra_late_p_values: str | Sequence[Decimal | float | int | str] = (),
) -> tuple[str, ...]:
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
    registry: dict[str, ReferenceClockSpec] = {}
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


def reference_clock_provenance(key: str) -> dict[str, Any]:
    normalized = str(key).strip().lower()
    base_key = normalized.removesuffix("_reversed")
    extras: tuple[Decimal, ...] = ()
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


def _normalize_descending(values: Sequence[float]) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not bool(np.all(np.isfinite(array))):
        raise ValueError("Reference nodes must be a finite one-dimensional sequence with at least two values.")
    if not bool(np.all(np.diff(array) < 0.0)):
        raise ValueError("Reference nodes must be strictly descending.")
    progression = (array[0] - array) / (array[0] - array[-1])
    progression[0], progression[-1] = 0.0, 1.0
    return tuple(float(value) for value in progression)


def _resample(progression: Sequence[float], n_steps: int) -> tuple[float, ...]:
    reference = np.asarray(progression, dtype=np.float64)
    src = np.linspace(0.0, 1.0, reference.size, dtype=np.float64)
    dst = np.linspace(0.0, 1.0, int(n_steps) + 1, dtype=np.float64)
    return _finalize(np.interp(dst, src, reference))


def _finalize(values: Sequence[float]) -> tuple[float, ...]:
    grid = np.asarray(values, dtype=np.float64).copy()
    if grid.ndim != 1 or grid.size < 2 or not bool(np.all(np.isfinite(grid))):
        raise ValueError("Reference clock grid must be finite and one-dimensional.")
    grid[0], grid[-1] = 0.0, 1.0
    if not bool(np.all(np.diff(grid) > 0.0)):
        raise ValueError("Reference clock grid must be strictly increasing.")
    return tuple(float(value) for value in grid)


def reverse_reference_clock_grid(grid: Sequence[float]) -> tuple[float, ...]:
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


def _ays_progression(coordinate: str) -> tuple[float, ...]:
    if coordinate == "native":
        return _normalize_descending((*AYS_SD15_TIMESTEPS, 0.0))
    positive_sigmas = (*AYS_SD15_SIGMAS[:-1], _sd15_sigma_ratio_t0())
    return _normalize_descending(tuple(math.log(value) for value in positive_sigmas))


def _gits_progression(coordinate: str) -> tuple[float, ...]:
    values = GITS_CIFAR10_SIGMAS if coordinate == "native" else tuple(math.log(value) for value in GITS_CIFAR10_SIGMAS)
    return _normalize_descending(values)


@cache
def ots_vp_linear_source_nodes(n_steps: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    n_steps = _validate_n_steps(n_steps)
    try:
        times = OTS_VP_LINEAR_OFFICIAL_TIMES[n_steps]
        lambdas = OTS_VP_LINEAR_OFFICIAL_LAMBDAS[n_steps]
    except KeyError as exc:
        raise ValueError(
            "Pinned official OTS nodes are available only for supported source step counts "
            f"{OTS_VP_LINEAR_SUPPORTED_STEP_COUNTS}; got {n_steps}."
        ) from exc
    return times, lambdas


def build_reference_clock_grid(key: str, n_steps: int) -> tuple[float, ...]:
    normalized = str(key).strip().lower()
    n_steps = _validate_n_steps(n_steps)
    base_key = normalized.removesuffix("_reversed")
    try:
        extra_late_p_values: tuple[Decimal, ...] = (
            (late_p_value_from_key(base_key),) if base_key.startswith("late_p_") else ()
        )
        registry = reference_clock_registry(extra_late_p_values)
    except (KeyError, ValueError) as exc:
        raise KeyError(f"Unknown reference clock {key!r}.") from exc
    if normalized not in registry:
        raise KeyError(f"Unknown reference clock {key!r}.")
    if normalized.endswith("_reversed"):
        return reverse_reference_clock_grid(build_reference_clock_grid(base_key, n_steps))
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
    "OTS_VP_LINEAR_OFFICIAL_LAMBDAS",
    "OTS_VP_LINEAR_OFFICIAL_TIMES",
    "OTS_VP_LINEAR_REALIZATION_ENVIRONMENT",
    "OTS_VP_LINEAR_SUPPORTED_STEP_COUNTS",
    "OTS_VP_LINEAR_TABLE_SHA256",
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
