from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from .registry import ImageBackboneSpec

if TYPE_CHECKING:
    from .protocol import ImageBackboneManifest


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}

IMAGE_BACKBONE_CONTEXT_SELECTOR = "native_model.model.map_label"
IMAGE_BACKBONE_CONTEXT_DIM = 768
_IMAGENET64_CLASS_COUNT = 1000


def _normalized_class_labels(
    labels: Tensor | None,
    *,
    spec: ImageBackboneSpec,
    batch_size: int,
    device: torch.device,
) -> Tensor | None:
    if spec.conditioning == "unconditional":
        if labels is not None:
            raise ValueError(f"Backbone {spec.key!r} is unconditional and rejects class labels.")
        return None
    if labels is None:
        raise ValueError(f"Backbone {spec.key!r} requires class labels.")
    if not isinstance(labels, Tensor):
        raise TypeError("Class labels must be a torch.Tensor.")

    num_classes = spec.num_conditioning_classes
    if labels.ndim == 1:
        if labels.shape[0] != batch_size:
            raise ValueError("Class-ID batch size does not match the image batch.")
        if labels.dtype not in _INTEGER_DTYPES:
            raise ValueError("Class IDs must use an integer tensor dtype.")
        if torch.any(labels < 0).item() or torch.any(labels >= num_classes).item():
            raise ValueError(f"Class IDs must be in [0, {num_classes}).")
        return torch.nn.functional.one_hot(
            labels.to(device=device, dtype=torch.int64),
            num_classes=num_classes,
        ).to(dtype=torch.float32)

    if labels.ndim != 2 or tuple(labels.shape) != (batch_size, num_classes):
        raise ValueError(f"One-hot labels must have shape ({batch_size}, {num_classes}).")
    if labels.is_complex():
        raise ValueError("One-hot labels must be real.")
    labels_on_device = labels.to(device=device)
    if not torch.isfinite(labels_on_device).all().item():
        raise ValueError("One-hot labels must be finite.")
    if not torch.all((labels_on_device == 0) | (labels_on_device == 1)).item():
        raise ValueError("One-hot labels must contain only exact zeros and ones.")
    if not torch.all(labels_on_device.sum(dim=1) == 1).item():
        raise ValueError("Each one-hot row must contain exactly one active class.")
    return labels_on_device.to(dtype=torch.float32)


class CanonicalNoiseToDataAdapter(nn.Module):
    """Expose an RF++ native velocity in the canonical noise-to-data coordinate."""

    def __init__(self, native_model: nn.Module, manifest: ImageBackboneManifest) -> None:
        super().__init__()
        self.native_model = native_model
        self.manifest = manifest

    def forward(self, state: Tensor, u: Tensor, labels: Tensor | None = None) -> Tensor:
        if not isinstance(state, Tensor):
            raise TypeError("Image state must be a torch.Tensor.")
        spec = self.manifest_spec
        expected_shape = spec.image_shape
        if state.ndim != 4 or tuple(state.shape[1:]) != expected_shape:
            raise ValueError(
                f"Image state must have shape [batch, {expected_shape[0]}, {expected_shape[1]}, {expected_shape[2]}]."
            )
        if not state.is_floating_point() or state.is_complex():
            raise ValueError("Image state must use a real floating-point dtype.")
        if not torch.isfinite(state).all().item():
            raise ValueError("Image state must be finite.")

        canonical_time = self._canonical_time(u, batch_size=state.shape[0], device=state.device)
        class_labels = _normalized_class_labels(
            labels,
            spec=spec,
            batch_size=state.shape[0],
            device=state.device,
        )
        epsilon = self.manifest.native_time_epsilon
        native_time = torch.clamp(1.0 - canonical_time, min=epsilon, max=1.0 - epsilon)
        native_velocity = self.native_model(state, native_time, class_labels=class_labels)
        if not isinstance(native_velocity, Tensor):
            raise TypeError("Native RF++ model must return a torch.Tensor.")
        if native_velocity.shape != state.shape:
            raise ValueError("Native RF++ velocity shape does not match the image state.")
        if native_velocity.device != state.device:
            raise ValueError("Native RF++ velocity must remain on the image-state device.")
        if not native_velocity.is_floating_point() or native_velocity.is_complex():
            raise ValueError("Native RF++ velocity must use a real floating-point dtype.")
        if not torch.isfinite(native_velocity).all().item():
            raise ValueError("Native RF++ velocity must be finite.")
        return -native_velocity

    def encode_conditioning(self, class_ids: Tensor) -> Tensor:
        """Encode ImageNet class IDs with the frozen native label projection.

        The returned representation is the class-only contribution used by the
        pinned RF++ DhariwalUNet before it is combined with its noise embedding.
        No image state, time, schedule, or stochastic input enters this path.
        """

        if not isinstance(class_ids, Tensor):
            raise TypeError("Class IDs must be a torch.Tensor.")
        if class_ids.layout != torch.strided or class_ids.device.type == "meta":
            raise ValueError("Class IDs must be a materialized strided tensor.")
        if class_ids.ndim != 1:
            raise ValueError("Class IDs for backbone conditioning must have shape [batch].")
        if class_ids.shape[0] == 0:
            raise ValueError("Class-ID batches must not be empty.")
        if class_ids.dtype not in _INTEGER_DTYPES:
            raise ValueError("Class IDs must use an integer tensor dtype.")
        if torch.any(class_ids < 0).item() or torch.any(class_ids >= _IMAGENET64_CLASS_COUNT).item():
            raise ValueError(f"Class IDs must be in [0, {_IMAGENET64_CLASS_COUNT}).")

        projection = self._conditioning_projection()
        weight = projection.weight
        one_hot = torch.nn.functional.one_hot(
            class_ids.to(device=weight.device, dtype=torch.int64),
            num_classes=_IMAGENET64_CLASS_COUNT,
        ).to(dtype=torch.float32)
        with torch.no_grad():
            encoded = projection(one_hot)
        if not isinstance(encoded, Tensor):
            raise TypeError("Native map_label must return a torch.Tensor.")
        expected_shape = (class_ids.shape[0], IMAGE_BACKBONE_CONTEXT_DIM)
        if tuple(encoded.shape) != expected_shape:
            raise ValueError(f"Native map_label output must have shape {expected_shape}.")
        if encoded.device != weight.device:
            raise ValueError("Native map_label output must remain on its projection device.")
        if encoded.dtype != torch.float32 or encoded.is_complex():
            raise ValueError("Native map_label output must use torch.float32.")
        if encoded.requires_grad or encoded.grad_fn is not None:
            raise RuntimeError("Native map_label output must be detached from autograd.")
        if not torch.isfinite(encoded).all().item():
            raise ValueError("Native map_label output must be finite.")
        return encoded.to(device=class_ids.device).detach().contiguous()

    def canonical_conditioning_table(self) -> Tensor:
        """Return ordered class-only contexts for ImageNet IDs 0 through 999."""

        table = self.encode_conditioning(torch.arange(_IMAGENET64_CLASS_COUNT, dtype=torch.int64, device="cpu"))
        expected_shape = (_IMAGENET64_CLASS_COUNT, IMAGE_BACKBONE_CONTEXT_DIM)
        if tuple(table.shape) != expected_shape or table.device.type != "cpu":
            raise RuntimeError("Canonical backbone conditioning table contract was violated.")
        if table.dtype != torch.float32 or table.requires_grad or not table.is_contiguous():
            raise RuntimeError("Canonical backbone conditioning table must be contiguous detached float32.")
        return table

    def _conditioning_projection(self) -> nn.Module:
        spec = self.manifest_spec
        if (
            spec.dataset_key != "imagenet64"
            or spec.conditioning != "class_conditional"
            or spec.num_conditioning_classes != _IMAGENET64_CLASS_COUNT
            or spec.architecture != "DhariwalUNet+EDMPrecondVel"
        ):
            raise ValueError(
                "Backbone conditioning contexts require the pinned class-conditional "
                "ImageNet-64 DhariwalUNet+EDMPrecondVel architecture."
            )
        if type(self.native_model).__name__ != "EDMPrecondVel":
            raise ValueError("Native ImageNet backbone must be the pinned EDMPrecondVel module.")
        unet = getattr(self.native_model, "model", None)
        if not isinstance(unet, nn.Module) or type(unet).__name__ != "DhariwalUNet":
            raise ValueError("Native EDMPrecondVel must contain the pinned DhariwalUNet module.")
        projection = getattr(unet, "map_label", None)
        if not isinstance(projection, nn.Module) or not callable(projection):
            raise ValueError("Native DhariwalUNet must expose a callable map_label module.")
        if type(getattr(projection, "in_features", None)) is not int or projection.in_features != 1000:
            raise ValueError("Native map_label in_features must equal 1000.")
        if type(getattr(projection, "out_features", None)) is not int or projection.out_features != 768:
            raise ValueError("Native map_label out_features must equal 768.")
        if not hasattr(projection, "bias") or projection.bias is not None:
            raise ValueError("Native map_label must be bias-free.")
        weight = getattr(projection, "weight", None)
        if not isinstance(weight, Tensor) or tuple(weight.shape) != (768, 1000):
            raise ValueError("Native map_label weight must have shape [768, 1000].")
        if weight.dtype != torch.float32 or weight.is_complex():
            raise ValueError("Native map_label weight must use torch.float32.")
        if not torch.isfinite(weight).all().item():
            raise ValueError("Native map_label weight must be finite.")
        if self.training or any(module.training for module in self.native_model.modules()):
            raise RuntimeError("Backbone conditioning encoding requires evaluation mode.")
        if any(parameter.requires_grad for parameter in self.native_model.parameters()):
            raise RuntimeError("Backbone conditioning encoding requires a frozen native model.")
        return projection

    @property
    def manifest_spec(self) -> ImageBackboneSpec:
        from .registry import get_image_backbone_spec

        return get_image_backbone_spec(self.manifest.model_key)

    @staticmethod
    def _canonical_time(u: Tensor, *, batch_size: int, device: torch.device) -> Tensor:
        if not isinstance(u, Tensor):
            raise TypeError("Canonical time u must be a torch.Tensor.")
        if not u.is_floating_point() or u.is_complex():
            raise ValueError("Canonical time u must use a real floating-point dtype.")
        if u.ndim == 0:
            normalized = u.expand(batch_size)
        elif u.ndim == 1 and u.shape[0] == batch_size:
            normalized = u
        elif u.ndim == 2 and tuple(u.shape) == (batch_size, 1):
            normalized = u[:, 0]
        else:
            raise ValueError("Canonical time u must be scalar, [batch], or [batch, 1].")
        normalized = normalized.to(device=device, dtype=torch.float32)
        if not torch.isfinite(normalized).all().item():
            raise ValueError("Canonical time u must be finite.")
        if torch.any(normalized < 0).item() or torch.any(normalized >= 1).item():
            raise ValueError("Canonical field evaluations require 0 <= u < 1.")
        return normalized


__all__ = [
    "CanonicalNoiseToDataAdapter",
    "IMAGE_BACKBONE_CONTEXT_DIM",
    "IMAGE_BACKBONE_CONTEXT_SELECTOR",
]
