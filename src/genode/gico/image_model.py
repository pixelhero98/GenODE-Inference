from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from genode.artifacts.identity import semantic_sha256
from genode.benchmarks.image.protocol import IMAGE_TARGET_NFES


IMAGE_GICO_MODEL_PROTOCOL = "image_gico_nfe_logits_v2"
DEFAULT_DENSITY_FLOOR = 1e-8


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be a positive integer, got {value!r}.")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field} must be positive, got {parsed}.")
    return parsed


def _density_floor(value: object, *, bin_count: int) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("density_floor must be a finite nonnegative real.")
    floor = float(value)
    if not math.isfinite(floor) or floor < 0.0:
        raise ValueError("density_floor must be finite and nonnegative.")
    if floor * float(bin_count) >= 1.0:
        raise ValueError("density_floor * density_bin_count must be strictly less than 1.")
    return floor


@dataclass(frozen=True)
class ImageGICOModelConfig:
    density_bin_count: int
    density_floor: float = DEFAULT_DENSITY_FLOOR
    target_nfes: tuple[int, ...] = IMAGE_TARGET_NFES

    def __post_init__(self) -> None:
        bin_count = _positive_integer(
            self.density_bin_count,
            field="density_bin_count",
        )
        target_nfes = tuple(self.target_nfes)
        if target_nfes != IMAGE_TARGET_NFES:
            raise ValueError(f"Image GICO model target_nfes must be exactly {IMAGE_TARGET_NFES} in canonical order.")
        object.__setattr__(self, "density_bin_count", bin_count)
        object.__setattr__(
            self,
            "density_floor",
            _density_floor(self.density_floor, bin_count=bin_count),
        )
        object.__setattr__(self, "target_nfes", target_nfes)

    def as_payload(self) -> dict[str, Any]:
        return {
            "protocol": IMAGE_GICO_MODEL_PROTOCOL,
            "density_bin_count": self.density_bin_count,
            "density_floor": self.density_floor,
            "target_nfes": list(self.target_nfes),
            "conditioning": "target_nfe_only",
        }

    @property
    def sha256(self) -> str:
        return semantic_sha256(
            self.as_payload(),
            namespace="image-gico-model-config",
        )

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> ImageGICOModelConfig:
        if not isinstance(payload, Mapping):
            raise TypeError("Image GICO model config must be a mapping.")
        expected_fields = {
            "protocol",
            "density_bin_count",
            "density_floor",
            "target_nfes",
            "conditioning",
        }
        if set(payload) != expected_fields:
            raise ValueError(f"Image GICO model config fields must be exactly {sorted(expected_fields)}.")
        if payload["protocol"] != IMAGE_GICO_MODEL_PROTOCOL:
            raise ValueError("Unsupported Image GICO model protocol.")
        if payload["conditioning"] != "target_nfe_only":
            raise ValueError("Image GICO model conditioning must be target_nfe_only.")
        raw_target_nfes = payload["target_nfes"]
        if not isinstance(raw_target_nfes, (list, tuple)):
            raise TypeError("target_nfes must be a sequence.")
        return cls(
            density_bin_count=payload["density_bin_count"],
            density_floor=payload["density_floor"],
            target_nfes=tuple(raw_target_nfes),
        )


class NFEConditionedDensityModel(nn.Module):
    """One distribution-level density policy conditioned only on target NFE."""

    def __init__(self, config: ImageGICOModelConfig) -> None:
        super().__init__()
        if not isinstance(config, ImageGICOModelConfig):
            raise TypeError("config must be an ImageGICOModelConfig.")
        self.config = config
        self.logits_by_nfe = nn.Parameter(
            torch.zeros(
                len(config.target_nfes),
                config.density_bin_count,
                dtype=torch.float32,
            )
        )

    def _nfe_indices(self, target_nfes: Tensor) -> Tensor:
        if not isinstance(target_nfes, Tensor):
            raise TypeError("target_nfes must be a torch.Tensor.")
        if target_nfes.ndim != 1 or int(target_nfes.numel()) <= 0:
            raise ValueError("target_nfes must have shape [batch].")
        if target_nfes.dtype == torch.bool or target_nfes.is_floating_point():
            raise TypeError("target_nfes must use an integer tensor dtype.")
        if target_nfes.device != self.logits_by_nfe.device:
            raise ValueError("target_nfes and Image GICO model must be on the same device.")
        supported = torch.tensor(
            self.config.target_nfes,
            dtype=target_nfes.dtype,
            device=target_nfes.device,
        )
        matches = target_nfes[:, None] == supported[None, :]
        if not bool(torch.all(matches.sum(dim=1) == 1)):
            observed = sorted({int(value) for value in target_nfes.detach().to(device="cpu").tolist()})
            raise ValueError(f"target_nfes must contain only {self.config.target_nfes}; got {observed}.")
        return torch.argmax(matches.to(dtype=torch.int64), dim=1)

    def forward(self, target_nfes: Tensor) -> Tensor:
        indices = self._nfe_indices(target_nfes)
        logits = self.logits_by_nfe[indices]
        probabilities = torch.softmax(logits, dim=-1)
        floor = self.config.density_floor
        if floor == 0.0:
            return probabilities
        free_mass = 1.0 - floor * float(self.config.density_bin_count)
        return floor + free_mass * probabilities


__all__ = [
    "DEFAULT_DENSITY_FLOOR",
    "IMAGE_GICO_MODEL_PROTOCOL",
    "ImageGICOModelConfig",
    "NFEConditionedDensityModel",
]
