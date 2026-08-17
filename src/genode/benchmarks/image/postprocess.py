from __future__ import annotations

import hashlib

import torch
from torch import Tensor

from genode.artifacts.identity import semantic_sha256

IMAGE_METRIC_POSTPROCESS_PROTOCOL = "image_edm_uint8_postprocess_v1"
IMAGE_METRIC_POSTPROCESS_SHA256 = semantic_sha256(
    {
        "protocol": IMAGE_METRIC_POSTPROCESS_PROTOCOL,
        "input": {
            "dtype": "float32",
            "layout": "nchw",
            "device": "cpu",
        },
        "expression": "(x * 127.5 + 128).clip(0,255).to(uint8)",
        "output": {
            "dtype": "uint8",
            "layout": "nchw_c_contiguous",
            "device": "cpu",
            "value_range": [0, 255],
        },
    },
    namespace="image-metric-postprocess",
)


def _raw_images(images: Tensor) -> Tensor:
    if not isinstance(images, Tensor):
        raise TypeError("images must be a torch.Tensor.")
    if images.ndim != 4 or int(images.shape[0]) <= 0:
        raise ValueError("Raw images must have shape [batch, channels, height, width].")
    if images.device.type != "cpu" or images.dtype != torch.float32 or not images.is_contiguous():
        raise ValueError("Raw metric images must be contiguous CPU torch.float32.")
    if not bool(torch.isfinite(images).all()):
        raise ValueError("Raw metric images must be finite.")
    return images


def metric_uint8_images(images: Tensor) -> Tensor:
    """Apply the pinned EDM/RF++ image conversion used by torch-fidelity."""

    raw = _raw_images(images)
    return (raw * 127.5 + 128.0).clamp(0.0, 255.0).to(dtype=torch.uint8).contiguous()


def metric_image_content_sha256(images: Tensor) -> str:
    if not isinstance(images, Tensor):
        raise TypeError("images must be a torch.Tensor.")
    if images.ndim != 4 or int(images.shape[0]) <= 0:
        raise ValueError("Metric images must have shape [batch, channels, height, width].")
    if images.device.type != "cpu" or images.dtype != torch.uint8 or not images.is_contiguous():
        raise ValueError("Metric images must be contiguous CPU torch.uint8.")
    return hashlib.sha256(images.numpy().tobytes(order="C")).hexdigest()


__all__ = [
    "IMAGE_METRIC_POSTPROCESS_PROTOCOL",
    "IMAGE_METRIC_POSTPROCESS_SHA256",
    "metric_image_content_sha256",
    "metric_uint8_images",
]
