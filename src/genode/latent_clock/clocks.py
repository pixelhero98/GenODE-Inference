from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genode.gico.clocks import REFERENCE_KEYS, materialize, reference_densities, validate_mass
from genode.gico.models import validate_time_grid

REFERENCE_CLOCK_KEYS = REFERENCE_KEYS
if (
    len(REFERENCE_CLOCK_KEYS) != 25
    or "late_p_3" not in REFERENCE_CLOCK_KEYS
    or "late_p_3_reversed" not in REFERENCE_CLOCK_KEYS
):
    raise RuntimeError("The latent-T2I reference support must contain 25 clocks including both late-p=3 directions.")


@dataclass(frozen=True)
class Clock:
    key: str
    target_nfe: int
    nodes: tuple[float, ...]
    source_kind: str = "genode_reference"
    density_mass: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        target = int(self.target_nfe)
        nodes = validate_time_grid(self.nodes, macro_steps=target)
        object.__setattr__(self, "target_nfe", target)
        object.__setattr__(self, "nodes", nodes)
        if not self.key or not self.source_kind:
            raise ValueError("Clock identities must be non-empty.")
        if self.density_mass is not None:
            mass = tuple(validate_mass(self.density_mass))
            if not np.array_equal(nodes, materialize(mass, "euler", target)):
                raise ValueError("Latent clock nodes differ from their shared 64-bin density realization.")
            object.__setattr__(self, "density_mass", mass)

    @property
    def widths(self) -> np.ndarray:
        return np.diff(np.asarray(self.nodes, dtype=np.float64))


def reference_clocks(target_nfe: int) -> tuple[Clock, ...]:
    densities = reference_densities("euler", int(target_nfe))
    return tuple(
        Clock(key=key, target_nfe=int(target_nfe), nodes=materialize(mass, "euler", int(target_nfe)), density_mass=mass)
        for key, mass in densities.items()
    )


PG_CLOCK_PRECISION = {
    "version": "sana-float32-spacing-v1",
    "minimum_width_on_collapse": 2.0**-20,
    "rule": "Keep representable clocks exactly; otherwise use delta + (1-K*delta)*softmax(logits).",
    "resampling": False,
}


def sana_grid_is_representable(nodes: np.ndarray) -> bool:
    sigmas = (1.0 - nodes).astype(np.float32)
    timesteps = sigmas[:-1] * np.float32(1000)
    return bool(np.all(np.diff(nodes) > 0) and np.all(np.diff(sigmas) < 0) and np.all(np.diff(timesteps) < 0))


def clock_from_interval_logits(key: str, logits: np.ndarray | list[float], *, pg_precision: bool = False) -> Clock:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Interval logits must be a non-empty finite vector.")
    anchored = np.concatenate([values, np.zeros(1, dtype=np.float64)])
    with np.errstate(over="ignore"):
        anchored -= np.max(anchored)
    widths = np.exp(anchored)
    widths /= widths.sum()
    nodes = np.concatenate([np.zeros(1), np.cumsum(widths)])
    nodes[-1] = 1.0
    corrected = False
    if pg_precision and not sana_grid_is_representable(nodes):
        # A deterministic action-to-clock map preserves PPO's log probability
        # of the original logistic-normal action; there is no rejection sampling.
        delta = PG_CLOCK_PRECISION["minimum_width_on_collapse"]
        if len(widths) * delta >= 1:
            raise ValueError("Too many intervals for the PG runtime spacing rule.")
        widths = delta + (1 - len(widths) * delta) * widths
        nodes = np.concatenate([np.zeros(1), np.cumsum(widths)])
        nodes[-1] = 1.0
        if not sana_grid_is_representable(nodes):
            raise RuntimeError("PG spacing rule failed to materialize a representable complete clock.")
        corrected = True
    return Clock(
        key=str(key),
        target_nfe=int(values.size + 1),
        nodes=tuple(float(x) for x in nodes),
        source_kind="pg_precision_corrected" if corrected else "genode_reference",
    )


def interval_logits(clock: Clock) -> np.ndarray:
    widths = clock.widths
    return np.log(widths[:-1]) - np.log(widths[-1])


def bo_bounds(target_nfe: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.stack([interval_logits(clock) for clock in reference_clocks(target_nfe)])
    return rows.min(axis=0) - 1.0, rows.max(axis=0) + 1.0
