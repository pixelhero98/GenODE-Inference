from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch

from genode.canonical_experiment_layout import CANONICAL_SEEN_NFES
from genode.solver_protocol import (
    CANONICAL_SOLVER_KEYS,
    normalize_solver_key,
    solver_effective_order,
    solver_eval_multiplier,
    solver_id_map,
    solver_macro_steps,
)

SOLVER_TO_ID: Dict[str, int] = solver_id_map(CANONICAL_SOLVER_KEYS)
TARGET_NFES: Tuple[int, ...] = CANONICAL_SEEN_NFES
DEFAULT_NFE_REFERENCE = 16
SETTING_ENCODER_MODE_CONTINUOUS_V3 = "continuous_v3"
SERIES_ENCODING_NONE_CONTEXT_ONLY = "none_context_only"
DEFAULT_SERIES_ENCODING = SERIES_ENCODING_NONE_CONTEXT_ONLY
SOLVER_METADATA_VERSION = "solver_metadata"


def validate_time_grid(grid: Sequence[float], *, macro_steps: int) -> Tuple[float, ...]:
    values = tuple(float(x) for x in grid)
    if len(values) != int(macro_steps) + 1:
        raise ValueError(f"Grid length {len(values)} does not match macro_steps={macro_steps}.")
    if abs(values[0]) > 1e-8 or abs(values[-1] - 1.0) > 1e-8:
        raise ValueError("Schedule grid must start at 0.0 and end at 1.0.")
    if not all(torch.isfinite(torch.tensor(values)).tolist()):
        raise ValueError("Schedule grid contains non-finite values.")
    if not all(b > a for a, b in zip(values, values[1:])):
        raise ValueError("Schedule grid must be strictly increasing.")
    return values


def validate_setting_feature_mode(mode: str) -> str:
    value = str(mode).strip() or SETTING_ENCODER_MODE_CONTINUOUS_V3
    allowed = {SETTING_ENCODER_MODE_CONTINUOUS_V3}
    if value not in allowed:
        raise ValueError(f"setting_feature_mode must be {SETTING_ENCODER_MODE_CONTINUOUS_V3!r}, got {mode!r}.")
    return value


def _hash_unit(text: str) -> float:
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    return float(int.from_bytes(digest[:8], "big") / float(2**64 - 1))


def _fourier_phase_features(phase: float, frequencies: Sequence[float]) -> Tuple[float, ...]:
    return tuple(
        component
        for frequency in frequencies
        for component in (math.sin(float(frequency) * float(phase)), math.cos(float(frequency) * float(phase)))
    )


def validate_setting_encoder_mode(mode: str) -> str:
    return validate_setting_feature_mode(mode)


def validate_series_encoding(value: str | None = None) -> str:
    encoding = str(value or SERIES_ENCODING_NONE_CONTEXT_ONLY).strip()
    if encoding != SERIES_ENCODING_NONE_CONTEXT_ONLY:
        raise ValueError(
            f"GIPO setting encoder requires series_encoding={SERIES_ENCODING_NONE_CONTEXT_ONLY!r}; "
            f"got {encoding!r}."
        )
    return encoding


def _positive_sorted_ints(values: Sequence[Any], *, label: str) -> Tuple[int, ...]:
    out = tuple(sorted({int(value) for value in values}))
    if not out or any(value <= 0 for value in out):
        raise ValueError(f"{label} must contain positive integer NFEs.")
    return out


@dataclass(frozen=True)
class SettingEncoderConfig:
    mode: str = SETTING_ENCODER_MODE_CONTINUOUS_V3
    observed_target_nfes: Tuple[int, ...] = TARGET_NFES
    nfe_reference: int = DEFAULT_NFE_REFERENCE
    rope_frequencies: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    solver_metadata_version: str = SOLVER_METADATA_VERSION
    series_encoding: str = DEFAULT_SERIES_ENCODING

    def to_payload(self) -> Dict[str, Any]:
        return {
            "mode": str(self.mode),
            "observed_target_nfes": [int(value) for value in self.observed_target_nfes],
            "nfe_reference": int(self.nfe_reference),
            "rope_frequencies": [float(value) for value in self.rope_frequencies],
            "solver_metadata_version": str(self.solver_metadata_version),
            "series_encoding": str(self.series_encoding),
        }


def build_setting_encoder_config(
    mode: str = SETTING_ENCODER_MODE_CONTINUOUS_V3,
    *,
    observed_target_nfes: Sequence[int] | None = None,
    nfe_reference: int | None = None,
    rope_frequencies: Sequence[float] | None = None,
    series_encoding: str = DEFAULT_SERIES_ENCODING,
    solver_metadata_version: str = SOLVER_METADATA_VERSION,
) -> SettingEncoderConfig:
    encoder_mode = validate_setting_encoder_mode(mode)
    observed = _positive_sorted_ints(TARGET_NFES if observed_target_nfes is None else observed_target_nfes, label="observed_target_nfes")
    reference = int(max(observed + (DEFAULT_NFE_REFERENCE,)) if nfe_reference is None else nfe_reference)
    if reference <= 0:
        raise ValueError("nfe_reference must be positive.")
    rope = tuple(float(value) for value in ((1.0, 2.0, 4.0, 8.0) if rope_frequencies is None else rope_frequencies))
    if not rope or any((not math.isfinite(value) or value <= 0.0) for value in rope):
        raise ValueError("rope_frequencies must contain finite positive values.")
    version = str(solver_metadata_version)
    if version != SOLVER_METADATA_VERSION:
        raise ValueError(f"Unsupported solver_metadata_version {version!r}; expected {SOLVER_METADATA_VERSION!r}.")
    return SettingEncoderConfig(
        mode=encoder_mode,
        observed_target_nfes=observed,
        nfe_reference=reference,
        rope_frequencies=rope,
        solver_metadata_version=version,
        series_encoding=validate_series_encoding(series_encoding),
    )


def setting_encoder_config_from_payload(payload: Mapping[str, Any] | SettingEncoderConfig | None) -> SettingEncoderConfig:
    if isinstance(payload, SettingEncoderConfig):
        validate_series_encoding(payload.series_encoding)
        return payload
    data = dict(payload or {})
    mode = str(data.get("mode", data.get("setting_feature_mode", SETTING_ENCODER_MODE_CONTINUOUS_V3)))
    return build_setting_encoder_config(
        mode,
        observed_target_nfes=data.get("observed_target_nfes", TARGET_NFES),
        nfe_reference=data.get("nfe_reference", DEFAULT_NFE_REFERENCE),
        rope_frequencies=data.get("rope_frequencies", None),
        series_encoding=str(data.get("series_encoding", DEFAULT_SERIES_ENCODING)),
        solver_metadata_version=str(data.get("solver_metadata_version", SOLVER_METADATA_VERSION)),
    )


def setting_encoder_config_for_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    series_encoding: str = DEFAULT_SERIES_ENCODING,
) -> SettingEncoderConfig:
    observed = sorted({int(row["target_nfe"]) for row in rows})
    return build_setting_encoder_config(mode, observed_target_nfes=observed or TARGET_NFES, series_encoding=series_encoding)


def setting_feature_dim(
    mode: str = SETTING_ENCODER_MODE_CONTINUOUS_V3,
    *,
    config: Mapping[str, Any] | SettingEncoderConfig | None = None,
) -> int:
    return int(setting_features("euler", 4, mode=mode, config=config).numel())


def _continuous_v3_features(solver_key: str, target_nfe: int, config: SettingEncoderConfig) -> torch.Tensor:
    solver = normalize_solver_key(str(solver_key))
    target = float(target_nfe)
    macro_steps = float(solver_macro_steps(solver, int(target_nfe)))
    reference = float(max(int(config.nfe_reference), int(max(config.observed_target_nfes)), int(target_nfe)))
    log_reference = math.log(max(reference, 1.000001))
    solver_phase = 2.0 * math.pi * _hash_unit(f"solver:{solver}")
    nfe_phase = math.pi * (math.log(max(target, 1.0)) / log_reference)
    below_flag = 1.0 if target < float(min(config.observed_target_nfes)) else 0.0
    above_flag = 1.0 if target > float(max(config.observed_target_nfes)) else 0.0
    features = [
        *_fourier_phase_features(solver_phase, (1.0, 2.0)),
        float(solver_effective_order(solver)) / 2.0,
        float(solver_eval_multiplier(solver)) / 2.0,
        math.log1p(macro_steps) / math.log1p(reference),
        math.log1p(target) / math.log1p(reference),
        below_flag,
        above_flag,
        *_fourier_phase_features(nfe_phase, config.rope_frequencies),
    ]
    return torch.tensor(features, dtype=torch.float32)


def setting_features(
    solver_key: str,
    target_nfe: int,
    *,
    mode: str = SETTING_ENCODER_MODE_CONTINUOUS_V3,
    config: Mapping[str, Any] | SettingEncoderConfig | None = None,
) -> torch.Tensor:
    feature_mode = validate_setting_feature_mode(mode)
    encoder_config = (
        setting_encoder_config_from_payload(config)
        if config is not None
        else build_setting_encoder_config(feature_mode)
    )
    requested_encoder_mode = validate_setting_encoder_mode(feature_mode)
    if requested_encoder_mode != encoder_config.mode:
        raise ValueError(
            f"setting_feature_mode {feature_mode!r} resolves to {requested_encoder_mode!r}, "
            f"but setting_encoder_config uses {encoder_config.mode!r}."
        )
    return _continuous_v3_features(str(solver_key), int(target_nfe), encoder_config)
