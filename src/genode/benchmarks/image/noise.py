from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from numbers import Integral

import numpy as np
import torch
from torch import Tensor

from genode.artifacts.identity import semantic_sha256
from genode.backbones.registry import get_image_dataset_spec


IMAGE_NOISE_PROTOCOL = "image_seeded_pcg64_normal_f32_v1"
IMAGE_NOISE_PROTOCOL_SHA256 = semantic_sha256(
    {
        "protocol": IMAGE_NOISE_PROTOCOL,
        "bit_generator": "numpy.random.PCG64",
        "distribution": "standard_normal",
        "stream_assignment": "one_independent_generator_per_sample_seed",
        "array_dtype": "float32",
        "array_layout": "cpu_c_contiguous_nchw",
        "seed_range": [0, 2**63 - 1],
    },
    namespace="image-noise-protocol",
)
_MAX_SEED = 2**63 - 1


def normalize_latent_seeds(values: Iterable[object]) -> tuple[int, ...]:
    seeds = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("latent seeds must be integers.")
        seed = int(value)
        if seed < 0 or seed > _MAX_SEED:
            raise ValueError(f"latent seeds must be in [0, {_MAX_SEED}].")
        seeds.append(seed)
    if not seeds:
        raise ValueError("At least one latent seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Latent seeds must be unique within one batch.")
    return tuple(seeds)


def _canonical_float32_bytes(tensor: Tensor) -> bytes:
    array = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return np.asarray(array, dtype="<f4", order="C").tobytes(order="C")


def image_tensor_content_sha256(tensor: Tensor) -> str:
    if not isinstance(tensor, Tensor):
        raise TypeError("tensor must be a torch.Tensor.")
    if tensor.ndim != 4 or int(tensor.shape[0]) <= 0:
        raise ValueError("Image tensor must have shape [batch, channels, height, width].")
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        raise ValueError("Portable image tensor hashing requires CPU torch.float32.")
    if not tensor.is_contiguous():
        raise ValueError("Portable image tensor hashing requires C-contiguous storage.")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("Image tensor must contain only finite values.")
    return hashlib.sha256(_canonical_float32_bytes(tensor)).hexdigest()


@dataclass(frozen=True)
class SeededImageNoiseBatch:
    dataset_key: str
    latent_seeds: tuple[int, ...]
    values: Tensor
    content_sha256: str = field(init=False)
    batch_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        dataset = get_image_dataset_spec(self.dataset_key)
        seeds = normalize_latent_seeds(self.latent_seeds)
        if not isinstance(self.values, Tensor):
            raise TypeError("noise values must be a torch.Tensor.")
        expected_shape = (len(seeds), *dataset.image_shape)
        if tuple(self.values.shape) != expected_shape:
            raise ValueError(f"Noise values have shape {tuple(self.values.shape)}; expected {expected_shape}.")
        if self.values.device.type != "cpu" or self.values.dtype != torch.float32 or not self.values.is_contiguous():
            raise ValueError("Noise values must be contiguous CPU torch.float32.")
        if not bool(torch.isfinite(self.values).all()):
            raise ValueError("Noise values must contain only finite values.")
        values = self.values.detach().clone(memory_format=torch.contiguous_format)
        content_sha256 = image_tensor_content_sha256(values)
        object.__setattr__(self, "dataset_key", dataset.key)
        object.__setattr__(self, "latent_seeds", seeds)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "content_sha256", content_sha256)
        object.__setattr__(
            self,
            "batch_sha256",
            semantic_sha256(
                self.identity_payload(),
                namespace="image-seeded-noise-batch",
            ),
        )

    @property
    def sample_count(self) -> int:
        return len(self.latent_seeds)

    def identity_payload(self) -> dict[str, object]:
        return {
            "protocol": IMAGE_NOISE_PROTOCOL,
            "protocol_sha256": IMAGE_NOISE_PROTOCOL_SHA256,
            "dataset_key": self.dataset_key,
            "latent_seeds": list(self.latent_seeds),
            "sample_count": self.sample_count,
            "shape": list(self.values.shape),
            "dtype": "float32",
            "layout": "cpu_c_contiguous_nchw",
            "content_sha256": self.content_sha256,
        }

    def as_payload(self) -> dict[str, object]:
        return {
            "artifact": "image_seeded_noise_batch",
            **self.identity_payload(),
            "batch_sha256": self.batch_sha256,
        }

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        values: Tensor,
    ) -> SeededImageNoiseBatch:
        if not isinstance(payload, Mapping):
            raise TypeError("Noise-batch payload must be a mapping.")
        expected_fields = {
            "artifact",
            "protocol",
            "protocol_sha256",
            "dataset_key",
            "latent_seeds",
            "sample_count",
            "shape",
            "dtype",
            "layout",
            "content_sha256",
            "batch_sha256",
        }
        if set(payload) != expected_fields:
            raise ValueError(f"Noise-batch fields must be exactly {sorted(expected_fields)}.")
        if (
            payload["artifact"] != "image_seeded_noise_batch"
            or payload["protocol"] != IMAGE_NOISE_PROTOCOL
            or payload["protocol_sha256"] != IMAGE_NOISE_PROTOCOL_SHA256
            or payload["dtype"] != "float32"
            or payload["layout"] != "cpu_c_contiguous_nchw"
        ):
            raise ValueError("Noise batch uses an incompatible protocol.")
        raw_seeds = payload["latent_seeds"]
        if not isinstance(raw_seeds, Sequence) or isinstance(
            raw_seeds,
            (str, bytes, bytearray),
        ):
            raise TypeError("latent_seeds must be a sequence.")
        batch = cls(
            dataset_key=payload["dataset_key"],  # type: ignore[arg-type]
            latent_seeds=tuple(raw_seeds),  # type: ignore[arg-type]
            values=values,
        )
        if (
            payload["sample_count"] != batch.sample_count
            or payload["shape"] != list(batch.values.shape)
            or payload["content_sha256"] != batch.content_sha256
            or payload["batch_sha256"] != batch.batch_sha256
        ):
            raise ValueError("Noise-batch payload does not match its tensor content.")
        return batch


def generate_seeded_image_noise(
    dataset_key: str,
    latent_seeds: Iterable[object],
) -> SeededImageNoiseBatch:
    """Generate one independent CPU float32 standard-normal image per seed."""

    dataset = get_image_dataset_spec(dataset_key)
    seeds = normalize_latent_seeds(latent_seeds)
    values = np.empty(
        (len(seeds), *dataset.image_shape),
        dtype=np.float32,
        order="C",
    )
    for index, seed in enumerate(seeds):
        generator = np.random.Generator(np.random.PCG64(seed))
        values[index] = generator.standard_normal(
            dataset.image_shape,
            dtype=np.float32,
        )
    return SeededImageNoiseBatch(
        dataset_key=dataset.key,
        latent_seeds=seeds,
        values=torch.from_numpy(values),
    )


__all__ = [
    "IMAGE_NOISE_PROTOCOL",
    "IMAGE_NOISE_PROTOCOL_SHA256",
    "SeededImageNoiseBatch",
    "generate_seeded_image_noise",
    "image_tensor_content_sha256",
    "normalize_latent_seeds",
]
