"""Device-local RNG scope helpers for image GICO training."""

from __future__ import annotations

import torch


def resolve_image_gico_training_device(
    device: torch.device | str,
) -> tuple[torch.device, list[int]]:
    """Resolve CPU or one exact CUDA device and its fork-RNG index list."""

    execution_device = torch.device(device)
    if execution_device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be CPU or CUDA.")
    if execution_device.type == "cpu":
        if execution_device.index is not None:
            raise ValueError("CPU training does not accept a device index.")
        return torch.device("cpu"), []
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable.")
    selected_index = torch.cuda.current_device() if execution_device.index is None else execution_device.index
    if not 0 <= selected_index < torch.cuda.device_count():
        raise ValueError("The selected CUDA device index is unavailable.")
    return torch.device("cuda", selected_index), [selected_index]


def seed_image_gico_training_generators(seed: int, execution_device: torch.device) -> None:
    """Seed CPU and, only when selected, the exact CUDA execution generator."""

    torch.random.default_generator.manual_seed(seed)
    if execution_device.type == "cuda":
        selected_index = execution_device.index
        if selected_index is None:
            raise ValueError("CUDA training must use one resolved device index.")
        torch.cuda.default_generators[selected_index].manual_seed(seed)


__all__ = [
    "resolve_image_gico_training_device",
    "seed_image_gico_training_generators",
]
