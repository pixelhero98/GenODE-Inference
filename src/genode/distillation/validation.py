from __future__ import annotations

from typing import Any, Mapping

from genode.checkpoint_validation import validate_strict_integer
from genode.gico.models import (
    SettingEncoderConfig,
    setting_encoder_config_from_payload as _load_setting_encoder_config,
)


def setting_encoder_config_from_payload(
    payload: Mapping[str, Any] | SettingEncoderConfig | None,
    *,
    require_complete: bool = False,
) -> SettingEncoderConfig:
    """Load a setting encoder without accepting omitted or unknown fields."""

    if isinstance(payload, SettingEncoderConfig):
        return _load_setting_encoder_config(payload)
    if payload is not None and not isinstance(payload, Mapping):
        raise ValueError("Setting encoder configuration must be an object.")
    data = dict(payload or {})
    expected_fields = set(_load_setting_encoder_config(None).to_payload())
    missing = sorted(expected_fields - set(data)) if require_complete else []
    unknown = sorted(set(data) - expected_fields)
    if missing or unknown:
        raise ValueError(
            "Setting encoder configuration fields are invalid; "
            f"missing={missing}, unknown={unknown}."
        )
    if "nfe_reference" in data:
        data["nfe_reference"] = validate_strict_integer(
            data["nfe_reference"],
            label="Setting encoder nfe_reference",
            minimum=1,
        )
    if "observed_target_nfes" in data:
        observed = data["observed_target_nfes"]
        if not isinstance(observed, (list, tuple)):
            raise ValueError("Setting encoder observed_target_nfes must be an integer sequence.")
        data["observed_target_nfes"] = [
            validate_strict_integer(
                value,
                label=f"Setting encoder observed_target_nfes[{index}]",
                minimum=1,
            )
            for index, value in enumerate(observed)
        ]
    return _load_setting_encoder_config(data)


__all__ = ["setting_encoder_config_from_payload"]
