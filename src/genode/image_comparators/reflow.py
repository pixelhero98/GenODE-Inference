"""Load the original ReFlow EMA network without changing its implementation."""

from __future__ import annotations

import importlib
import sys
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

_IMPORT_LOCK = threading.RLock()
RF_START = 1e-4
RF_END = 0.9999


@contextmanager
def _upstream_imports(root: Path):
    """Scope RF's absolute imports so BézierFlow's ``models`` is not overwritten.

    Only loading runs in this context; the NCSN++ forward pass uses already bound
    modules. Call during single-threaded runtime initialization.
    """
    names = ("models", "op", "sde_lib")

    def owned(name):
        return any(name == prefix or name.startswith(prefix + ".") for prefix in names)

    with _IMPORT_LOCK:
        previous = {name: module for name, module in sys.modules.items() if owned(name)}
        previous_path = sys.path.copy()
        for name in previous:
            del sys.modules[name]
        sys.path.insert(0, str(root))
        try:
            yield
        finally:
            for name in list(sys.modules):
                if owned(name):
                    del sys.modules[name]
            sys.modules.update(previous)
            sys.path[:] = previous_path


def _namespace(value):
    if isinstance(value, Mapping):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    return value


def restore_ema(net: nn.Module, state: Mapping) -> None:
    """Strictly restore parameters and buffers, then the original EMA ordering."""
    net.load_state_dict(state["model"], strict=True)
    parameters = [parameter for parameter in net.parameters() if parameter.requires_grad]
    shadows = state["ema"]["shadow_params"]
    if len(parameters) != len(shadows):
        raise ValueError("Checkpoint EMA tensor count differs from the ReFlow network")
    for parameter, shadow in zip(parameters, shadows, strict=True):
        if parameter.shape != shadow.shape or not torch.isfinite(shadow).all():
            raise ValueError("Checkpoint EMA shape or finite-value validation failed")
    with torch.no_grad():
        for parameter, shadow in zip(parameters, shadows, strict=True):
            parameter.copy_(shadow)


def load_reflow(source: str | Path, checkpoint: str | Path, config: Mapping, *, device: str) -> nn.Module:
    """Use the pinned RectifiedFlow checkout and caller-verified checkpoint.

    ``source`` is the repository root. Config is the upstream CIFAR ReFlow model
    mapping (e.g. BézierFlow's cifar10_rf_gaussian_ddpmpp.yml after YAML parsing).
    CUDA build dependencies for the original operators must be installed.
    """
    root = Path(source).resolve() / "ImageGeneration"
    if not (root / "models" / "ncsnpp.py").is_file():
        raise ValueError(f"Not a RectifiedFlow source checkout: {source}")
    configuration = _namespace(config)
    if configuration.model.name != "ncsnpp":
        raise ValueError("This bridge supports only the original CIFAR NCSN++ network")
    configuration.device = torch.device(device)
    with _upstream_imports(root):
        importlib.import_module("models.ncsnpp")
        utilities = importlib.import_module("models.utils")
        net = utilities.create_model(configuration)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restore_ema(net, state)
    net.eval().requires_grad_(False)
    return net


class ReflowVelocity(nn.Module):
    """BézierFlow's ReFlow time convention, with actual network-call accounting."""

    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network
        self.calls = 0

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.network(x, time.expand(x.shape[0]) * 1000).float()

    def bezier(self, x, _solver_time, model_time, *_conditioning):
        """Signature expected by upstream BézierFlow solvers/converters."""
        return self(x, model_time)


def reflow_grid(progress, *, device) -> torch.Tensor:
    """Map normalized clock progress onto the upstream RF sampling horizon."""
    progress = torch.as_tensor(progress, dtype=torch.float64, device=device)
    if (
        progress.ndim != 1
        or len(progress) < 2
        or not torch.isfinite(progress).all()
        or progress[0] != 0
        or progress[-1] != 1
        or not (progress.diff() > 0).all()
    ):
        raise ValueError("Clock must be a finite, strictly increasing vector from zero to one")
    grid = (RF_START + (RF_END - RF_START) * progress).float()
    if not (grid.diff() > 0).all():
        raise ValueError("ReFlow grid collapses in the executed float32 time representation")
    return grid


def euler_sample(velocity: ReflowVelocity, noise: torch.Tensor, progress) -> torch.Tensor:
    """Preserve the field and use exactly one network call per Euler interval."""
    grid = reflow_grid(progress, device=noise.device)
    before = velocity.calls
    x = noise
    for left, right in zip(grid[:-1], grid[1:], strict=True):
        x = x + (right - left) * velocity(x, left)
    if velocity.calls - before != len(grid) - 1:
        raise RuntimeError("Executed ReFlow network calls do not match the requested NFE")
    return x
