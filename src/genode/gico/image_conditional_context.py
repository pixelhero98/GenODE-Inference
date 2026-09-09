"""Native image-backbone contexts; normalization belongs to common GICO."""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from genode.backbones.adapter import CanonicalNoiseToDataAdapter
from genode.backbones.protocol import ImageBackboneManifest
from genode.backbones.registry import get_image_backbone_spec


def context_binding(manifest: ImageBackboneManifest, native_context_table) -> dict:
    task = get_image_backbone_spec(manifest.model_key).dataset_key
    table = np.asarray(native_context_table, dtype="<f4")
    expected_shape = (1000, 768) if task == "imagenet64" else (1, 1)
    if table.shape != expected_shape or not np.isfinite(table).all():
        raise ValueError(f"Native {task} context table must have finite shape {expected_shape}.")
    if task == "cifar10" and np.any(table != 0):
        raise ValueError("CIFAR-10 policy context must be explicitly zero.")
    return {
        "model_key": manifest.model_key,
        "protocol_sha256": manifest.protocol_sha256,
        "checkpoint_sha256": manifest.checkpoint.sha256,
        "context_table_sha256": hashlib.sha256(np.ascontiguousarray(table).tobytes()).hexdigest(),
        "context_shape": list(table.shape),
        "context_source": "native_class_embedding" if task == "imagenet64" else "zero",
    }


def native_contexts(backbone: CanonicalNoiseToDataAdapter) -> tuple[np.ndarray, dict]:
    if not isinstance(backbone, CanonicalNoiseToDataAdapter):
        raise TypeError("Native context binding requires the verified backbone adapter.")
    if backbone.training or any(parameter.requires_grad for parameter in backbone.parameters()):
        raise ValueError("Native context binding requires a frozen evaluation backbone.")
    task = get_image_backbone_spec(backbone.manifest.model_key).dataset_key
    with torch.no_grad():
        table = (
            backbone.canonical_conditioning_table().cpu().numpy()
            if task == "imagenet64"
            else np.zeros((1, 1), dtype=np.float32)
        )
    return table, context_binding(backbone.manifest, table)
