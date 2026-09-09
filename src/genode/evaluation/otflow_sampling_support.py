from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def _sample_cfg_snapshot(cfg) -> dict[str, Any]:
    return dict(cfg.to_dict()["sample"])


def _apply_sample_overrides(model: torch.nn.Module, cfg, **overrides: Any) -> dict[str, Any]:
    backup = _sample_cfg_snapshot(cfg)
    clean = {key: value for key, value in overrides.items() if value is not None}
    if clean:
        cfg.apply_overrides(**clean)
        if getattr(model, "cfg", None) is not cfg:
            model.cfg.apply_overrides(**clean)
    return backup


def _restore_sample_overrides(model: torch.nn.Module, cfg, backup: Mapping[str, Any]) -> None:
    cfg.apply_overrides(**dict(backup))
    if getattr(model, "cfg", None) is not cfg:
        model.cfg.apply_overrides(**dict(backup))


__all__ = ["_apply_sample_overrides", "_restore_sample_overrides"]
