from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import math
from numbers import Integral, Real
import re
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from genode.artifacts.identity import semantic_sha256
from genode.backbones.adapter import CanonicalNoiseToDataAdapter
from genode.backbones.protocol import ImageBackboneManifest
from genode.backbones.registry import get_image_backbone_spec
from genode.benchmarks.image.noise import (
    SeededImageNoiseBatch,
    generate_seeded_image_noise,
    image_tensor_content_sha256,
    normalize_latent_seeds,
)
from genode.benchmarks.image.postprocess import (
    IMAGE_METRIC_POSTPROCESS_PROTOCOL,
    IMAGE_METRIC_POSTPROCESS_SHA256,
    metric_image_content_sha256,
    metric_uint8_images,
)
from genode.benchmarks.image.protocol import normalize_image_nfe
from genode.schedule_transfer.reference_clocks import reference_clock_provenance
from genode.schedules.density import density_mass_hash, time_grid_hash
from genode.schedules.fixed import FixedSchedule
from genode.schedules.policy import (
    IdentifiedSchedulePolicy,
    ScheduleBatch,
    SchedulePolicy,
)
from genode.solvers.euler import integrate_euler

if TYPE_CHECKING:
    from genode.gico.image_clock_mixture_artifacts import (
        BoundImageGICOClockMixtureArtifact,
    )
    from genode.gico.image_conditional_artifacts import (
        BoundImageGICOConditionalArtifact,
    )


IMAGE_GENERATION_REQUEST_PROTOCOL = "image_euler_generation_request_v2"
IMAGE_GICO_CLOCK_REALIZATION_REQUEST_PROTOCOL = "image_euler_generation_request_v3"
IMAGE_GICO_CLOCK_REALIZATION_BINDING_PROTOCOL = (
    "image_gico_clock_realization_binding_v1"
)
IMAGE_GENERATED_BATCH_PROTOCOL = "image_generated_batch_v2"
IMAGE_GENERATED_BATCH_PROVENANCE_PROTOCOL = "image_generated_batch_provenance_v1"
IMAGE_EULER_EXECUTION_DTYPE = "float32"
_SHA256_IDENTITY = re.compile(r"(?:[a-z][a-z0-9_.-]*:)?[0-9a-f]{64}\Z")
_RAW_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _sha256_identity(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_IDENTITY.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 identity.")
    return value


def _raw_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _RAW_SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be one lowercase SHA-256 digest.")
    return value


def _clock_uniforms_sha256(uniforms: Tensor) -> str:
    """Hash the exact caller-supplied uniforms used for a clock realization."""

    values = uniforms.detach().to(device="cpu").contiguous().numpy()
    return semantic_sha256(
        {
            "dtype": str(values.dtype),
            "shape": [int(value) for value in values.shape],
            "content_sha256": hashlib.sha256(
                values.tobytes(order="C")
            ).hexdigest(),
        },
        namespace="image-gico-clock-uniforms-v1",
    )


def execution_time_grid_sha256(time_grid: Tensor) -> str:
    """Hash the exact CPU float32 grid consumed by the Euler runtime."""

    if not isinstance(time_grid, Tensor):
        raise TypeError("time_grid must be a torch.Tensor.")
    grid = time_grid.detach().to(
        device="cpu",
        dtype=torch.float32,
    )
    return time_grid_hash(grid)


def _class_labels(
    values: Sequence[object] | None,
    *,
    backbone_manifest: ImageBackboneManifest,
    sample_count: int,
) -> tuple[int, ...] | None:
    spec = get_image_backbone_spec(backbone_manifest.model_key)
    if spec.conditioning == "unconditional":
        if values is not None:
            raise ValueError(f"Backbone {spec.key!r} is unconditional and rejects labels.")
        return None
    if values is None or len(values) != sample_count:
        raise ValueError("Class-conditional generation requires one label per sample.")
    labels = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("Class labels must be integers.")
        label = int(value)
        if label < 0 or label >= spec.num_conditioning_classes:
            raise ValueError("Class label is outside the backbone conditioning range.")
        labels.append(label)
    return tuple(labels)


@dataclass(frozen=True)
class ImageGICOClockRealizationBinding:
    """Canonical replay metadata for one stochastic complete-clock draw."""

    realization_sha256: str
    schedule_sha256: str
    alpha: float
    selected_schedule_indices: tuple[int | None, ...]
    selected_schedule_keys: tuple[str | None, ...]
    probability_sha256: str
    uniforms_sha256: str | None
    clock_library_sha256: str
    policy_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, Real):
            raise TypeError("alpha must be a finite real in [0, 1].")
        alpha = float(self.alpha)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be a finite real in [0, 1].")
        indices = tuple(self.selected_schedule_indices)
        keys = tuple(self.selected_schedule_keys)
        if not indices or len(keys) != len(indices):
            raise ValueError(
                "Selected clock indices and keys must be nonempty and aligned."
            )
        normalized_indices: list[int | None] = []
        normalized_keys: list[str | None] = []
        if alpha == 0.0:
            if (
                any(index is not None for index in indices)
                or any(key is not None for key in keys)
                or self.uniforms_sha256 is not None
            ):
                raise ValueError(
                    "alpha=0 clock bindings must not depend on uniforms or selections."
                )
            normalized_indices.extend(None for _ in indices)
            normalized_keys.extend(None for _ in keys)
            uniforms_sha256 = None
        else:
            for index, key in zip(indices, keys, strict=True):
                if (
                    isinstance(index, bool)
                    or not isinstance(index, Integral)
                    or int(index) < 0
                ):
                    raise ValueError(
                        "Selected schedule indices must be nonnegative integers."
                    )
                if not isinstance(key, str):
                    raise TypeError(
                        "Selected schedule keys must be fixed-schedule keys."
                    )
                try:
                    reference_clock_provenance(key)
                except (KeyError, ValueError) as exc:
                    raise ValueError(
                        "Selected schedule keys must be supported fixed-schedule keys."
                    ) from exc
                normalized_indices.append(int(index))
                normalized_keys.append(key)
            uniforms_sha256 = _sha256_identity(
                self.uniforms_sha256,
                field="uniforms_sha256",
            )
        for field_name in (
            "realization_sha256",
            "schedule_sha256",
            "probability_sha256",
            "clock_library_sha256",
            "policy_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _sha256_identity(getattr(self, field_name), field=field_name),
            )
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(
            self,
            "selected_schedule_indices",
            tuple(normalized_indices),
        )
        object.__setattr__(
            self,
            "selected_schedule_keys",
            tuple(normalized_keys),
        )
        object.__setattr__(self, "uniforms_sha256", uniforms_sha256)
        expected_realization_sha256 = semantic_sha256(
            self.realization_identity_payload(),
            namespace="image-gico-clock-realization-v1",
        )
        if self.realization_sha256 != expected_realization_sha256:
            raise ValueError(
                "Clock realization identity is inconsistent with its replay metadata."
            )

    @property
    def sample_count(self) -> int:
        return len(self.selected_schedule_indices)

    def realization_identity_payload(self) -> dict[str, object]:
        from genode.gico.image_clock_mixture import (
            IMAGE_GICO_CLOCK_MIXTURE_POLICY_PROTOCOL,
        )

        return {
            "protocol": IMAGE_GICO_CLOCK_MIXTURE_POLICY_PROTOCOL,
            "schedule_sha256": self.schedule_sha256,
            "alpha": self.alpha,
            "selected_schedule_indices": list(self.selected_schedule_indices),
            "selected_schedule_keys": list(self.selected_schedule_keys),
            "probability_sha256": self.probability_sha256,
            "uniforms_sha256": self.uniforms_sha256,
            "clock_library_sha256": self.clock_library_sha256,
            "policy_sha256": self.policy_sha256,
        }

    def identity_payload(self) -> dict[str, object]:
        return {
            "protocol": IMAGE_GICO_CLOCK_REALIZATION_BINDING_PROTOCOL,
            "realization_sha256": self.realization_sha256,
            "schedule_sha256": self.schedule_sha256,
            "alpha": self.alpha,
            "selected_schedule_indices": list(self.selected_schedule_indices),
            "selected_schedule_keys": list(self.selected_schedule_keys),
            "probability_sha256": self.probability_sha256,
            "uniforms_sha256": self.uniforms_sha256,
            "clock_library_sha256": self.clock_library_sha256,
            "policy_sha256": self.policy_sha256,
        }

    def as_payload(self) -> dict[str, object]:
        return {
            "artifact": "image_gico_clock_realization_binding",
            **self.identity_payload(),
        }

    @classmethod
    def from_payload(
        cls,
        payload: object,
    ) -> ImageGICOClockRealizationBinding:
        if not isinstance(payload, Mapping):
            raise TypeError("Clock realization binding payload must be a mapping.")
        expected_fields = {
            "artifact",
            "protocol",
            "realization_sha256",
            "schedule_sha256",
            "alpha",
            "selected_schedule_indices",
            "selected_schedule_keys",
            "probability_sha256",
            "uniforms_sha256",
            "clock_library_sha256",
            "policy_sha256",
        }
        if set(payload) != expected_fields:
            raise ValueError(
                "Clock realization binding fields must be exactly "
                f"{sorted(expected_fields)}."
            )
        if (
            payload["artifact"] != "image_gico_clock_realization_binding"
            or payload["protocol"]
            != IMAGE_GICO_CLOCK_REALIZATION_BINDING_PROTOCOL
        ):
            raise ValueError("Unsupported clock realization binding artifact.")
        if type(payload["alpha"]) is not float:
            raise TypeError(
                "Clock realization binding alpha must be a JSON floating-point number."
            )
        raw_indices = payload["selected_schedule_indices"]
        raw_keys = payload["selected_schedule_keys"]
        if not isinstance(raw_indices, Sequence) or isinstance(
            raw_indices,
            (str, bytes, bytearray),
        ):
            raise TypeError("selected_schedule_indices must be a sequence.")
        if not isinstance(raw_keys, Sequence) or isinstance(
            raw_keys,
            (str, bytes, bytearray),
        ):
            raise TypeError("selected_schedule_keys must be a sequence.")
        binding = cls(
            realization_sha256=payload["realization_sha256"],  # type: ignore[arg-type]
            schedule_sha256=payload["schedule_sha256"],  # type: ignore[arg-type]
            alpha=payload["alpha"],  # type: ignore[arg-type]
            selected_schedule_indices=tuple(raw_indices),  # type: ignore[arg-type]
            selected_schedule_keys=tuple(raw_keys),  # type: ignore[arg-type]
            probability_sha256=payload["probability_sha256"],  # type: ignore[arg-type]
            uniforms_sha256=payload["uniforms_sha256"],  # type: ignore[arg-type]
            clock_library_sha256=payload["clock_library_sha256"],  # type: ignore[arg-type]
            policy_sha256=payload["policy_sha256"],  # type: ignore[arg-type]
        )
        if dict(payload) != binding.as_payload():
            raise ValueError("Clock realization binding payload is inconsistent.")
        return binding

    @classmethod
    def from_realization(
        cls,
        realization: object,
    ) -> ImageGICOClockRealizationBinding:
        from genode.gico.image_clock_mixture import ImageGICOClockRealization

        if type(realization) is not ImageGICOClockRealization:
            raise TypeError(
                "realization must be a canonical ImageGICOClockRealization."
            )
        return cls(
            realization_sha256=realization.sha256,
            schedule_sha256=realization.schedule.sha256,
            alpha=realization.alpha,
            selected_schedule_indices=realization.selected_schedule_indices,
            selected_schedule_keys=realization.selected_schedule_keys,
            probability_sha256=realization.probability_sha256,
            uniforms_sha256=realization.uniforms_sha256,
            clock_library_sha256=realization.clock_library_sha256,
            policy_sha256=realization.policy_sha256,
        )


@dataclass(frozen=True)
class ImageGenerationRequest:
    source_request_sha256: str
    backbone_manifest: ImageBackboneManifest
    latent_seeds: tuple[int, ...]
    class_labels: tuple[int, ...] | None
    target_nfe: int
    schedule_policy_sha256: str
    schedule_output_sha256: str
    time_grid_sha256: str
    execution_time_grid_sha256: str
    density_mass_sha256: str
    execution_time_grid_dtype: str = IMAGE_EULER_EXECUTION_DTYPE
    clock_realization: ImageGICOClockRealizationBinding | None = None
    request_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        source_request_sha256 = _sha256_identity(
            self.source_request_sha256,
            field="source_request_sha256",
        )
        if not isinstance(self.backbone_manifest, ImageBackboneManifest):
            raise TypeError("backbone_manifest must be an ImageBackboneManifest.")
        seeds = normalize_latent_seeds(self.latent_seeds)
        labels = _class_labels(
            self.class_labels,
            backbone_manifest=self.backbone_manifest,
            sample_count=len(seeds),
        )
        target_nfe = normalize_image_nfe(self.target_nfe)
        hashes = {
            field_name: _sha256_identity(
                getattr(self, field_name),
                field=field_name,
            )
            for field_name in (
                "schedule_policy_sha256",
                "schedule_output_sha256",
                "time_grid_sha256",
                "execution_time_grid_sha256",
                "density_mass_sha256",
            )
        }
        if self.execution_time_grid_dtype != IMAGE_EULER_EXECUTION_DTYPE:
            raise ValueError("Image Euler execution time grids must use float32.")
        if self.clock_realization is not None:
            if type(self.clock_realization) is not ImageGICOClockRealizationBinding:
                raise TypeError(
                    "clock_realization must be an ImageGICOClockRealizationBinding."
                )
            if self.clock_realization.sample_count != len(seeds):
                raise ValueError(
                    "Clock realization metadata must match the request batch size."
                )
            if labels is None:
                raise ValueError(
                    "Clock realization requests require class-conditional labels."
                )
        object.__setattr__(
            self,
            "source_request_sha256",
            source_request_sha256,
        )
        object.__setattr__(self, "latent_seeds", seeds)
        object.__setattr__(self, "class_labels", labels)
        object.__setattr__(self, "target_nfe", target_nfe)
        for field_name, value in hashes.items():
            object.__setattr__(self, field_name, value)
        object.__setattr__(
            self,
            "request_sha256",
            semantic_sha256(
                self.identity_payload(),
                namespace="image-euler-generation-request",
            ),
        )

    @property
    def dataset_key(self) -> str:
        return get_image_backbone_spec(self.backbone_manifest.model_key).dataset_key

    @property
    def sample_count(self) -> int:
        return len(self.latent_seeds)

    def identity_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "protocol": (
                IMAGE_GENERATION_REQUEST_PROTOCOL
                if self.clock_realization is None
                else IMAGE_GICO_CLOCK_REALIZATION_REQUEST_PROTOCOL
            ),
            "source_request_sha256": self.source_request_sha256,
            "dataset_key": self.dataset_key,
            "backbone_model_key": self.backbone_manifest.model_key,
            "backbone_protocol_sha256": (self.backbone_manifest.protocol_sha256),
            "backbone_checkpoint_sha256": (self.backbone_manifest.checkpoint.sha256),
            "latent_seeds": list(self.latent_seeds),
            "class_labels": (None if self.class_labels is None else list(self.class_labels)),
            "sample_count": self.sample_count,
            "solver_key": "euler",
            "target_nfe": self.target_nfe,
            "schedule_policy_sha256": self.schedule_policy_sha256,
            "schedule_output_sha256": self.schedule_output_sha256,
            "analytic_time_grid_sha256": self.time_grid_sha256,
            "execution_time_grid_dtype": self.execution_time_grid_dtype,
            "execution_time_grid_sha256": (self.execution_time_grid_sha256),
            "density_mass_sha256": self.density_mass_sha256,
        }
        if self.clock_realization is not None:
            payload["clock_realization"] = self.clock_realization.as_payload()
        return payload

    def as_payload(self) -> dict[str, object]:
        return {
            "artifact": "image_euler_generation_request",
            "identity": self.identity_payload(),
            "backbone_manifest": (self.backbone_manifest.to_manifest_dict()),
            "request_sha256": self.request_sha256,
        }

    @classmethod
    def from_payload(cls, payload: object) -> ImageGenerationRequest:
        if not isinstance(payload, Mapping):
            raise TypeError("Image generation request payload must be a mapping.")
        expected_fields = {
            "artifact",
            "identity",
            "backbone_manifest",
            "request_sha256",
        }
        if set(payload) != expected_fields:
            raise ValueError(f"Image generation request fields must be exactly {sorted(expected_fields)}.")
        if payload["artifact"] != "image_euler_generation_request":
            raise ValueError("Unsupported image generation request artifact.")
        identity = payload["identity"]
        if not isinstance(identity, Mapping):
            raise TypeError("Image generation request identity must be a mapping.")
        expected_identity_fields = {
            "protocol",
            "source_request_sha256",
            "dataset_key",
            "backbone_model_key",
            "backbone_protocol_sha256",
            "backbone_checkpoint_sha256",
            "latent_seeds",
            "class_labels",
            "sample_count",
            "solver_key",
            "target_nfe",
            "schedule_policy_sha256",
            "schedule_output_sha256",
            "analytic_time_grid_sha256",
            "execution_time_grid_dtype",
            "execution_time_grid_sha256",
            "density_mass_sha256",
        }
        protocol = identity.get("protocol")
        clock_realization = None
        if protocol == IMAGE_GICO_CLOCK_REALIZATION_REQUEST_PROTOCOL:
            expected_identity_fields.add("clock_realization")
            clock_realization = ImageGICOClockRealizationBinding.from_payload(
                identity.get("clock_realization")
            )
        elif protocol != IMAGE_GENERATION_REQUEST_PROTOCOL:
            raise ValueError("Image generation request uses an incompatible protocol.")
        if set(identity) != expected_identity_fields:
            raise ValueError(
                f"Image generation request identity fields must be exactly {sorted(expected_identity_fields)}."
            )
        if identity["solver_key"] != "euler":
            raise ValueError("Image generation request uses an incompatible protocol.")
        raw_seeds = identity["latent_seeds"]
        raw_labels = identity["class_labels"]
        if not isinstance(raw_seeds, Sequence) or isinstance(
            raw_seeds,
            (str, bytes, bytearray),
        ):
            raise TypeError("latent_seeds must be a sequence.")
        if raw_labels is not None and (
            not isinstance(raw_labels, Sequence) or isinstance(raw_labels, (str, bytes, bytearray))
        ):
            raise TypeError("class_labels must be null or a sequence.")
        request = cls(
            source_request_sha256=identity["source_request_sha256"],  # type: ignore[arg-type]
            backbone_manifest=ImageBackboneManifest.from_manifest_dict(payload["backbone_manifest"]),
            latent_seeds=tuple(raw_seeds),  # type: ignore[arg-type]
            class_labels=(
                None if raw_labels is None else tuple(raw_labels)  # type: ignore[arg-type]
            ),
            target_nfe=identity["target_nfe"],  # type: ignore[arg-type]
            schedule_policy_sha256=identity["schedule_policy_sha256"],  # type: ignore[arg-type]
            schedule_output_sha256=identity["schedule_output_sha256"],  # type: ignore[arg-type]
            time_grid_sha256=identity["analytic_time_grid_sha256"],  # type: ignore[arg-type]
            execution_time_grid_sha256=identity["execution_time_grid_sha256"],  # type: ignore[arg-type]
            density_mass_sha256=identity["density_mass_sha256"],  # type: ignore[arg-type]
            execution_time_grid_dtype=identity["execution_time_grid_dtype"],  # type: ignore[arg-type]
            clock_realization=clock_realization,
        )
        if dict(identity) != request.identity_payload() or payload["request_sha256"] != request.request_sha256:
            raise ValueError("Image generation request payload is inconsistent.")
        return request


@dataclass(frozen=True)
class ScheduleExecutionBinding:
    source_kind: str
    schedule_policy_sha256: str
    schedule_output_sha256: str
    time_grid_sha256: str
    execution_time_grid_sha256: str
    density_mass_sha256: str
    target_nfe: int
    sample_count: int
    execution_time_grid_dtype: str = IMAGE_EULER_EXECUTION_DTYPE
    clock_realization: ImageGICOClockRealizationBinding | None = None
    binding_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.source_kind not in {
            "fixed_schedule",
            "schedule_policy",
            "contextual_schedule_policy",
        }:
            raise ValueError("source_kind must be fixed_schedule, schedule_policy, or contextual_schedule_policy.")
        for field_name in (
            "schedule_policy_sha256",
            "schedule_output_sha256",
            "time_grid_sha256",
            "execution_time_grid_sha256",
            "density_mass_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _sha256_identity(
                    getattr(self, field_name),
                    field=field_name,
                ),
            )
        if self.execution_time_grid_dtype != IMAGE_EULER_EXECUTION_DTYPE:
            raise ValueError("Image Euler execution time grids must use float32.")
        target_nfe = normalize_image_nfe(self.target_nfe)
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count <= 0:
            raise ValueError("sample_count must be a positive integer.")
        if self.clock_realization is not None:
            if type(self.clock_realization) is not ImageGICOClockRealizationBinding:
                raise TypeError(
                    "clock_realization must be an ImageGICOClockRealizationBinding."
                )
            if self.source_kind == "fixed_schedule":
                raise ValueError(
                    "Fixed-schedule execution cannot carry a stochastic clock realization."
                )
            if self.clock_realization.sample_count != self.sample_count:
                raise ValueError(
                    "Clock realization metadata must match the execution batch size."
                )
        object.__setattr__(self, "target_nfe", target_nfe)
        object.__setattr__(
            self,
            "binding_sha256",
            semantic_sha256(
                self.identity_payload(),
                namespace="image-schedule-execution",
            ),
        )

    def identity_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source_kind": self.source_kind,
            "schedule_policy_sha256": self.schedule_policy_sha256,
            "schedule_output_sha256": self.schedule_output_sha256,
            "analytic_time_grid_sha256": self.time_grid_sha256,
            "execution_time_grid_dtype": self.execution_time_grid_dtype,
            "execution_time_grid_sha256": (self.execution_time_grid_sha256),
            "density_mass_sha256": self.density_mass_sha256,
            "solver_key": "euler",
            "target_nfe": self.target_nfe,
            "sample_count": self.sample_count,
        }
        if self.clock_realization is not None:
            payload["clock_realization"] = self.clock_realization.as_payload()
        return payload

    def as_payload(self) -> dict[str, object]:
        return {
            "artifact": "image_schedule_execution",
            **self.identity_payload(),
            "binding_sha256": self.binding_sha256,
        }

    @classmethod
    def from_payload(cls, payload: object) -> ScheduleExecutionBinding:
        if not isinstance(payload, Mapping):
            raise TypeError("Schedule execution payload must be a mapping.")
        expected_fields = {
            "artifact",
            "source_kind",
            "schedule_policy_sha256",
            "schedule_output_sha256",
            "analytic_time_grid_sha256",
            "execution_time_grid_dtype",
            "execution_time_grid_sha256",
            "density_mass_sha256",
            "solver_key",
            "target_nfe",
            "sample_count",
            "binding_sha256",
        }
        clock_realization = None
        if "clock_realization" in payload:
            expected_fields.add("clock_realization")
            clock_realization = ImageGICOClockRealizationBinding.from_payload(
                payload["clock_realization"]
            )
        if set(payload) != expected_fields:
            raise ValueError(f"Schedule execution fields must be exactly {sorted(expected_fields)}.")
        if payload["artifact"] != "image_schedule_execution" or payload["solver_key"] != "euler":
            raise ValueError("Unsupported schedule execution artifact.")
        binding = cls(
            source_kind=payload["source_kind"],  # type: ignore[arg-type]
            schedule_policy_sha256=payload["schedule_policy_sha256"],  # type: ignore[arg-type]
            schedule_output_sha256=payload["schedule_output_sha256"],  # type: ignore[arg-type]
            time_grid_sha256=payload["analytic_time_grid_sha256"],  # type: ignore[arg-type]
            execution_time_grid_sha256=payload["execution_time_grid_sha256"],  # type: ignore[arg-type]
            density_mass_sha256=payload["density_mass_sha256"],  # type: ignore[arg-type]
            target_nfe=payload["target_nfe"],  # type: ignore[arg-type]
            sample_count=payload["sample_count"],  # type: ignore[arg-type]
            execution_time_grid_dtype=payload["execution_time_grid_dtype"],  # type: ignore[arg-type]
            clock_realization=clock_realization,
        )
        if payload["binding_sha256"] != binding.binding_sha256:
            raise ValueError("Schedule execution hash is inconsistent.")
        return binding


@dataclass(frozen=True)
class GeneratedBatchProvenance:
    """Content-addressed generated-batch metadata retained by KID evidence."""

    request: ImageGenerationRequest
    noise_batch_sha256: str
    schedule: ScheduleExecutionBinding
    field_evaluations: int
    shape: tuple[int, ...]
    content_sha256: str
    metric_shape: tuple[int, ...]
    metric_content_sha256: str
    batch_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.request, ImageGenerationRequest):
            raise TypeError("request must be an ImageGenerationRequest.")
        noise_hash = _sha256_identity(
            self.noise_batch_sha256,
            field="noise_batch_sha256",
        )
        if not isinstance(self.schedule, ScheduleExecutionBinding):
            raise TypeError("schedule must be a ScheduleExecutionBinding.")
        expected_schedule = (
            self.request.schedule_policy_sha256,
            self.request.schedule_output_sha256,
            self.request.time_grid_sha256,
            self.request.execution_time_grid_sha256,
            self.request.execution_time_grid_dtype,
            self.request.density_mass_sha256,
            self.request.target_nfe,
            self.request.sample_count,
            self.request.clock_realization,
        )
        actual_schedule = (
            self.schedule.schedule_policy_sha256,
            self.schedule.schedule_output_sha256,
            self.schedule.time_grid_sha256,
            self.schedule.execution_time_grid_sha256,
            self.schedule.execution_time_grid_dtype,
            self.schedule.density_mass_sha256,
            self.schedule.target_nfe,
            self.schedule.sample_count,
            self.schedule.clock_realization,
        )
        if actual_schedule != expected_schedule:
            raise ValueError("Generated-batch provenance schedule does not match its request.")
        evaluations = self.field_evaluations
        if isinstance(evaluations, bool) or not isinstance(evaluations, int) or evaluations != self.request.target_nfe:
            raise ValueError("Generated-batch provenance field evaluations must equal target_nfe.")
        expected_shape = (
            self.request.sample_count,
            *get_image_backbone_spec(self.request.backbone_manifest.model_key).image_shape,
        )
        shape = tuple(self.shape)
        metric_shape = tuple(self.metric_shape)
        if shape != expected_shape or metric_shape != expected_shape:
            raise ValueError("Generated-batch provenance image shapes do not match its request.")
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in (*shape, *metric_shape)):
            raise TypeError("Generated-batch provenance shapes must contain integers.")
        content_hash = _raw_sha256(
            self.content_sha256,
            field="content_sha256",
        )
        metric_hash = _raw_sha256(
            self.metric_content_sha256,
            field="metric_content_sha256",
        )
        object.__setattr__(self, "noise_batch_sha256", noise_hash)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "metric_shape", metric_shape)
        object.__setattr__(self, "content_sha256", content_hash)
        object.__setattr__(
            self,
            "metric_content_sha256",
            metric_hash,
        )
        object.__setattr__(
            self,
            "batch_sha256",
            semantic_sha256(
                self.generated_batch_identity_payload(),
                namespace="image-generated-batch",
            ),
        )

    def generated_batch_identity_payload(self) -> dict[str, object]:
        return {
            "protocol": IMAGE_GENERATED_BATCH_PROTOCOL,
            "request_sha256": self.request.request_sha256,
            "noise_batch_sha256": self.noise_batch_sha256,
            "schedule_binding_sha256": self.schedule.binding_sha256,
            "field_evaluations": self.field_evaluations,
            "shape": list(self.shape),
            "dtype": "float32",
            "layout": "cpu_c_contiguous_nchw",
            "content_sha256": self.content_sha256,
            "metric_postprocess_protocol": (IMAGE_METRIC_POSTPROCESS_PROTOCOL),
            "metric_postprocess_sha256": (IMAGE_METRIC_POSTPROCESS_SHA256),
            "metric_shape": list(self.metric_shape),
            "metric_dtype": "uint8",
            "metric_layout": "cpu_c_contiguous_nchw",
            "metric_content_sha256": self.metric_content_sha256,
        }

    def identity_payload(self) -> dict[str, object]:
        return {
            "protocol": IMAGE_GENERATED_BATCH_PROVENANCE_PROTOCOL,
            "generated_batch": self.generated_batch_identity_payload(),
            "generated_batch_sha256": self.batch_sha256,
        }

    @property
    def provenance_sha256(self) -> str:
        return semantic_sha256(
            self.identity_payload(),
            namespace="image-generated-batch-provenance",
        )

    def as_payload(self) -> dict[str, object]:
        return {
            "artifact": "image_generated_batch_provenance",
            **self.identity_payload(),
            "request": self.request.as_payload(),
            "schedule": self.schedule.as_payload(),
            "provenance_sha256": self.provenance_sha256,
        }

    @classmethod
    def from_payload(cls, payload: object) -> GeneratedBatchProvenance:
        if not isinstance(payload, Mapping):
            raise TypeError("Generated-batch provenance payload must be a mapping.")
        expected_fields = {
            "artifact",
            "protocol",
            "generated_batch",
            "generated_batch_sha256",
            "request",
            "schedule",
            "provenance_sha256",
        }
        if set(payload) != expected_fields:
            raise ValueError(f"Generated-batch provenance fields must be exactly {sorted(expected_fields)}.")
        if (
            payload["artifact"] != "image_generated_batch_provenance"
            or payload["protocol"] != IMAGE_GENERATED_BATCH_PROVENANCE_PROTOCOL
        ):
            raise ValueError("Unsupported generated-batch provenance artifact.")
        identity = payload["generated_batch"]
        if not isinstance(identity, Mapping):
            raise TypeError("Generated-batch provenance identity must be a mapping.")
        expected_identity_fields = {
            "protocol",
            "request_sha256",
            "noise_batch_sha256",
            "schedule_binding_sha256",
            "field_evaluations",
            "shape",
            "dtype",
            "layout",
            "content_sha256",
            "metric_postprocess_protocol",
            "metric_postprocess_sha256",
            "metric_shape",
            "metric_dtype",
            "metric_layout",
            "metric_content_sha256",
        }
        if set(identity) != expected_identity_fields:
            raise ValueError(f"Generated-batch identity fields must be exactly {sorted(expected_identity_fields)}.")
        shape = identity["shape"]
        metric_shape = identity["metric_shape"]
        if not isinstance(shape, Sequence) or isinstance(
            shape,
            (str, bytes, bytearray),
        ):
            raise TypeError("Generated-batch shape must be a sequence.")
        if not isinstance(metric_shape, Sequence) or isinstance(
            metric_shape,
            (str, bytes, bytearray),
        ):
            raise TypeError("Generated-batch metric_shape must be a sequence.")
        provenance = cls(
            request=ImageGenerationRequest.from_payload(payload["request"]),
            noise_batch_sha256=identity["noise_batch_sha256"],  # type: ignore[arg-type]
            schedule=ScheduleExecutionBinding.from_payload(payload["schedule"]),
            field_evaluations=identity["field_evaluations"],  # type: ignore[arg-type]
            shape=tuple(shape),  # type: ignore[arg-type]
            content_sha256=identity["content_sha256"],  # type: ignore[arg-type]
            metric_shape=tuple(metric_shape),  # type: ignore[arg-type]
            metric_content_sha256=identity["metric_content_sha256"],  # type: ignore[arg-type]
        )
        if dict(identity) != provenance.generated_batch_identity_payload():
            raise ValueError("Generated-batch provenance identity is inconsistent.")
        if payload["generated_batch_sha256"] != provenance.batch_sha256:
            raise ValueError("Generated-batch provenance batch hash is inconsistent.")
        if payload["provenance_sha256"] != provenance.provenance_sha256:
            raise ValueError("Generated-batch provenance hash is inconsistent.")
        return provenance


@dataclass(frozen=True)
class GeneratedImageBatch:
    request: ImageGenerationRequest
    noise: SeededImageNoiseBatch
    schedule: ScheduleExecutionBinding
    images: Tensor
    field_evaluations: int
    content_sha256: str = field(init=False)
    metric_images: Tensor = field(init=False, repr=False)
    metric_content_sha256: str = field(init=False)
    batch_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.request, ImageGenerationRequest):
            raise TypeError("request must be an ImageGenerationRequest.")
        if not isinstance(self.noise, SeededImageNoiseBatch):
            raise TypeError("noise must be a SeededImageNoiseBatch.")
        if not isinstance(self.schedule, ScheduleExecutionBinding):
            raise TypeError("schedule must be a ScheduleExecutionBinding.")
        if self.noise.dataset_key != self.request.dataset_key or self.noise.latent_seeds != self.request.latent_seeds:
            raise ValueError("Generated batch noise does not match its request.")
        expected_schedule = (
            self.request.schedule_policy_sha256,
            self.request.schedule_output_sha256,
            self.request.time_grid_sha256,
            self.request.execution_time_grid_sha256,
            self.request.execution_time_grid_dtype,
            self.request.density_mass_sha256,
            self.request.target_nfe,
            self.request.sample_count,
            self.request.clock_realization,
        )
        actual_schedule = (
            self.schedule.schedule_policy_sha256,
            self.schedule.schedule_output_sha256,
            self.schedule.time_grid_sha256,
            self.schedule.execution_time_grid_sha256,
            self.schedule.execution_time_grid_dtype,
            self.schedule.density_mass_sha256,
            self.schedule.target_nfe,
            self.schedule.sample_count,
            self.schedule.clock_realization,
        )
        if actual_schedule != expected_schedule:
            raise ValueError("Generated batch schedule does not match its request.")
        if not isinstance(self.images, Tensor):
            raise TypeError("images must be a torch.Tensor.")
        image_shape = get_image_backbone_spec(self.request.backbone_manifest.model_key).image_shape
        expected_shape = (self.request.sample_count, *image_shape)
        if tuple(self.images.shape) != expected_shape:
            raise ValueError(f"Generated images have shape {tuple(self.images.shape)}; expected {expected_shape}.")
        if self.images.device.type != "cpu" or self.images.dtype != torch.float32 or not self.images.is_contiguous():
            raise ValueError("Generated images must be contiguous CPU torch.float32.")
        if not bool(torch.isfinite(self.images).all()):
            raise ValueError("Generated images must contain only finite values.")
        if (
            isinstance(self.field_evaluations, bool)
            or not isinstance(self.field_evaluations, int)
            or self.field_evaluations != self.request.target_nfe
        ):
            raise ValueError("Generated Euler field-evaluation count must equal target_nfe.")
        images = self.images.detach().clone(memory_format=torch.contiguous_format)
        content_sha256 = image_tensor_content_sha256(images)
        metric_images = metric_uint8_images(images)
        metric_content_sha256 = metric_image_content_sha256(metric_images)
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "content_sha256", content_sha256)
        object.__setattr__(self, "metric_images", metric_images)
        object.__setattr__(
            self,
            "metric_content_sha256",
            metric_content_sha256,
        )
        object.__setattr__(
            self,
            "batch_sha256",
            semantic_sha256(
                self.identity_payload(),
                namespace="image-generated-batch",
            ),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "protocol": IMAGE_GENERATED_BATCH_PROTOCOL,
            "request_sha256": self.request.request_sha256,
            "noise_batch_sha256": self.noise.batch_sha256,
            "schedule_binding_sha256": self.schedule.binding_sha256,
            "field_evaluations": self.field_evaluations,
            "shape": list(self.images.shape),
            "dtype": "float32",
            "layout": "cpu_c_contiguous_nchw",
            "content_sha256": self.content_sha256,
            "metric_postprocess_protocol": (IMAGE_METRIC_POSTPROCESS_PROTOCOL),
            "metric_postprocess_sha256": (IMAGE_METRIC_POSTPROCESS_SHA256),
            "metric_shape": list(self.metric_images.shape),
            "metric_dtype": "uint8",
            "metric_layout": "cpu_c_contiguous_nchw",
            "metric_content_sha256": self.metric_content_sha256,
        }

    def as_payload(self) -> dict[str, object]:
        return {
            "artifact": "image_generated_batch",
            "request": self.request.as_payload(),
            "noise": self.noise.as_payload(),
            "schedule": self.schedule.as_payload(),
            **self.identity_payload(),
            "batch_sha256": self.batch_sha256,
        }

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        images: Tensor,
    ) -> GeneratedImageBatch:
        if not isinstance(payload, Mapping):
            raise TypeError("Generated image batch payload must be a mapping.")
        expected_fields = {
            "artifact",
            "request",
            "noise",
            "schedule",
            "protocol",
            "request_sha256",
            "noise_batch_sha256",
            "schedule_binding_sha256",
            "field_evaluations",
            "shape",
            "dtype",
            "layout",
            "content_sha256",
            "metric_postprocess_protocol",
            "metric_postprocess_sha256",
            "metric_shape",
            "metric_dtype",
            "metric_layout",
            "metric_content_sha256",
            "batch_sha256",
        }
        if set(payload) != expected_fields:
            raise ValueError(f"Generated image batch fields must be exactly {sorted(expected_fields)}.")
        if (
            payload["artifact"] != "image_generated_batch"
            or payload["protocol"] != IMAGE_GENERATED_BATCH_PROTOCOL
            or payload["dtype"] != "float32"
            or payload["layout"] != "cpu_c_contiguous_nchw"
            or payload["metric_postprocess_protocol"] != IMAGE_METRIC_POSTPROCESS_PROTOCOL
            or payload["metric_postprocess_sha256"] != IMAGE_METRIC_POSTPROCESS_SHA256
            or payload["metric_dtype"] != "uint8"
            or payload["metric_layout"] != "cpu_c_contiguous_nchw"
        ):
            raise ValueError("Generated image batch uses an incompatible protocol.")
        request = ImageGenerationRequest.from_payload(payload["request"])
        noise = generate_seeded_image_noise(
            request.dataset_key,
            request.latent_seeds,
        )
        if payload["noise"] != noise.as_payload():
            raise ValueError("Generated image batch noise payload is inconsistent.")
        schedule = ScheduleExecutionBinding.from_payload(payload["schedule"])
        batch = cls(
            request=request,
            noise=noise,
            schedule=schedule,
            images=images,
            field_evaluations=payload["field_evaluations"],  # type: ignore[arg-type]
        )
        identity = batch.identity_payload()
        for field_name, value in identity.items():
            if payload[field_name] != value:
                raise ValueError("Generated image batch identity is inconsistent.")
        if payload["batch_sha256"] != batch.batch_sha256:
            raise ValueError("Generated image batch hash is inconsistent.")
        return batch


def _schedule_rows_are_shared(schedule: ScheduleBatch) -> bool:
    first_density = schedule.density_mass[0]
    first_grid = schedule.time_grid[0]
    return schedule.batch_size == 1 or (
        torch.equal(
            schedule.density_mass,
            first_density.unsqueeze(0).expand_as(schedule.density_mass),
        )
        and torch.equal(
            schedule.time_grid,
            first_grid.unsqueeze(0).expand_as(schedule.time_grid),
        )
    )


def policy_schedule_request_hashes(
    schedule: ScheduleBatch,
    *,
    preserve_batch: bool = False,
) -> tuple[str, str, str, str]:
    """Return request hashes for either a shared or contextual policy batch.

    Global policies retain the historical single-row identities. Callers that
    define a contextual protocol set ``preserve_batch=True`` to bind the
    complete ordered schedule batch, including the degenerate case where all
    rows currently happen to be identical.
    """

    if not isinstance(schedule, ScheduleBatch):
        raise TypeError("schedule must be a ScheduleBatch.")
    if not isinstance(preserve_batch, bool):
        raise TypeError("preserve_batch must be a boolean.")
    if _schedule_rows_are_shared(schedule) and not preserve_batch:
        first_density = schedule.density_mass[0]
        first_grid = schedule.time_grid[0]
        single = ScheduleBatch(
            density_mass=first_density.unsqueeze(0),
            reference_time_grid=schedule.reference_time_grid,
            time_grid=first_grid.unsqueeze(0),
            target_nfe=schedule.target_nfe,
            specification=schedule.specification,
        )
        return (
            single.sha256,
            time_grid_hash(first_grid),
            execution_time_grid_sha256(first_grid),
            density_mass_hash(
                first_density,
                reference_time_grid=schedule.reference_time_grid,
            ),
        )
    return (
        schedule.sha256,
        time_grid_hash(schedule.time_grid),
        execution_time_grid_sha256(schedule.time_grid),
        density_mass_hash(
            schedule.density_mass,
            reference_time_grid=schedule.reference_time_grid,
        ),
    )


def _policy_output_binding(
    schedule: ScheduleBatch,
    *,
    request: ImageGenerationRequest,
    source_kind: str,
    schedule_policy_sha256: str,
    clock_realization: ImageGICOClockRealizationBinding | None = None,
) -> ScheduleExecutionBinding:
    verified_policy_sha256 = _sha256_identity(
        schedule_policy_sha256,
        field="schedule_policy_sha256",
    )
    if verified_policy_sha256 != request.schedule_policy_sha256:
        raise ValueError("Schedule policy identity does not match the generation request.")
    if schedule.target_nfe != request.target_nfe:
        raise ValueError("Schedule output target_nfe does not match the request.")
    if schedule.batch_size != request.sample_count:
        raise ValueError("Schedule output batch size does not match the request.")
    if request.clock_realization != clock_realization:
        raise ValueError(
            "Schedule clock-realization binding does not match the generation request."
        )
    if (
        clock_realization is not None
        and clock_realization.schedule_sha256 != schedule.sha256
    ):
        raise ValueError(
            "Clock realization schedule identity does not match the executable schedule."
        )
    expected = (
        request.schedule_output_sha256,
        request.time_grid_sha256,
        request.execution_time_grid_sha256,
        request.density_mass_sha256,
    )
    shared = _schedule_rows_are_shared(schedule)
    actual = policy_schedule_request_hashes(schedule)
    batch_preserved = False
    if actual != expected and shared and schedule.batch_size > 1:
        full_batch = policy_schedule_request_hashes(
            schedule,
            preserve_batch=True,
        )
        if full_batch == expected:
            actual = full_batch
            batch_preserved = True
    if actual != expected:
        raise ValueError("Executable schedule output does not match the request's output/grid/density hashes.")
    return ScheduleExecutionBinding(
        source_kind=(source_kind if shared and not batch_preserved else "contextual_schedule_policy"),
        schedule_policy_sha256=verified_policy_sha256,
        schedule_output_sha256=actual[0],
        time_grid_sha256=actual[1],
        execution_time_grid_sha256=actual[2],
        density_mass_sha256=actual[3],
        target_nfe=request.target_nfe,
        sample_count=request.sample_count,
        clock_realization=clock_realization,
    )


def _fixed_schedule_grid(
    schedule: FixedSchedule,
    *,
    request: ImageGenerationRequest,
    device: torch.device,
) -> tuple[Tensor, ScheduleExecutionBinding]:
    """Bind a fixed schedule without quantizing its executable grid.

    ``FixedSchedule.density_mass`` is the finite-bin learning representation
    used by GICO.  Its ``time_grid`` is the exact analytical schedule used for
    integration.  The two are intentionally hash-bound but a finite-bin
    projection is not required to invert exactly to the analytical grid.
    Learned policies still use ``ScheduleBatch`` and its strict
    density-to-quantile executable binding.
    """

    if not isinstance(schedule, FixedSchedule):
        raise TypeError("schedule must be a FixedSchedule.")
    expected = (
        schedule.sha256,
        schedule.sha256,
        schedule.time_grid_sha256,
        execution_time_grid_sha256(schedule.time_grid),
        schedule.density_mass_sha256,
        schedule.target_nfe,
    )
    requested = (
        request.schedule_policy_sha256,
        request.schedule_output_sha256,
        request.time_grid_sha256,
        request.execution_time_grid_sha256,
        request.density_mass_sha256,
        request.target_nfe,
    )
    if expected != requested:
        raise ValueError("Fixed schedule does not match the generation request.")
    batch_size = request.sample_count
    grid = schedule.time_grid.to(device=device).unsqueeze(0).expand(batch_size, -1).clone()
    binding = ScheduleExecutionBinding(
        source_kind="fixed_schedule",
        schedule_policy_sha256=schedule.sha256,
        schedule_output_sha256=schedule.sha256,
        time_grid_sha256=schedule.time_grid_sha256,
        execution_time_grid_sha256=execution_time_grid_sha256(schedule.time_grid),
        density_mass_sha256=schedule.density_mass_sha256,
        target_nfe=schedule.target_nfe,
        sample_count=batch_size,
    )
    return grid, binding


class ImageEulerSampler:
    """Execute content-bound Euler generation through one verified adapter."""

    def __init__(
        self,
        backbone: CanonicalNoiseToDataAdapter,
        *,
        device: torch.device | str,
        execution_batch_size: int = 64,
    ) -> None:
        if not isinstance(backbone, CanonicalNoiseToDataAdapter):
            raise TypeError(
                "backbone must be a CanonicalNoiseToDataAdapter returned by the verified image-backbone loader."
            )
        execution_device = torch.device(device)
        if execution_device.type not in {"cpu", "cuda"}:
            raise ValueError("Image Euler execution device must be CPU or CUDA.")
        if execution_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA image execution was requested but CUDA is unavailable.")
            if execution_device.index is None:
                execution_device = torch.device(
                    "cuda",
                    torch.cuda.current_device(),
                )
        if backbone.training:
            raise ValueError("Image backbone must be in evaluation mode.")
        if any(parameter.requires_grad for parameter in backbone.parameters()):
            raise ValueError("Image backbone parameters must have gradients disabled.")
        if (
            isinstance(execution_batch_size, bool)
            or not isinstance(execution_batch_size, Integral)
            or int(execution_batch_size) <= 0
        ):
            raise ValueError("execution_batch_size must be a positive integer.")
        self.backbone = backbone.to(device=execution_device)
        self.device = execution_device
        self.execution_batch_size = int(execution_batch_size)

    def _validate_request(
        self,
        request: ImageGenerationRequest,
        *,
        allow_clock_realization: bool = False,
    ) -> None:
        if not isinstance(request, ImageGenerationRequest):
            raise TypeError("request must be an ImageGenerationRequest.")
        if not isinstance(allow_clock_realization, bool):
            raise TypeError("allow_clock_realization must be a boolean.")
        if request.clock_realization is not None and not allow_clock_realization:
            raise ValueError(
                "A realization-bound request must use sample_gico_clock_realization."
            )
        if request.backbone_manifest.to_manifest_dict() != self.backbone.manifest.to_manifest_dict():
            raise ValueError("Generation request does not match the loaded backbone.")

    def _bound_gico_clock_labels(
        self,
        request: ImageGenerationRequest,
        artifact: "BoundImageGICOClockMixtureArtifact",
        *,
        allow_clock_realization: bool = False,
    ) -> Tensor:
        from genode.gico.image_clock_mixture_artifacts import (
            BoundImageGICOClockMixtureArtifact,
        )

        self._validate_request(
            request,
            allow_clock_realization=allow_clock_realization,
        )
        if not isinstance(artifact, BoundImageGICOClockMixtureArtifact):
            raise TypeError(
                "artifact must be a bound ImageNet GICO clock-mixture artifact."
            )
        if request.class_labels is None:
            raise ValueError(
                "Bound ImageNet GICO clock-mixture execution requires class labels."
            )
        if artifact.artifact_sha256 != request.schedule_policy_sha256:
            raise ValueError(
                "Bound GICO clock-mixture artifact identity does not match "
                "the generation request."
            )
        binding = artifact.source_artifact.prepared_context.binding
        expected_backbone = (
            request.backbone_manifest.model_key,
            request.backbone_manifest.protocol_sha256,
            request.backbone_manifest.checkpoint.sha256,
        )
        observed_backbone = (
            binding.backbone_model_key,
            binding.backbone_protocol_sha256,
            binding.backbone_checkpoint_sha256,
        )
        if observed_backbone != expected_backbone:
            raise ValueError(
                "Bound GICO clock-mixture context does not match the "
                "generation-request backbone."
            )
        artifact.verify_execution_identity()
        return torch.tensor(request.class_labels, dtype=torch.int64)

    def _execute(
        self,
        request: ImageGenerationRequest,
        *,
        noise: SeededImageNoiseBatch,
        time_grid: Tensor,
        schedule_binding: ScheduleExecutionBinding,
    ) -> GeneratedImageBatch:
        grid = time_grid.to(
            device="cpu",
            dtype=torch.float32,
        )
        if grid.shape != (request.sample_count, request.target_nfe + 1):
            raise ValueError("Executable time grid shape does not match the generation request.")
        contextual = schedule_binding.source_kind == "contextual_schedule_policy"
        observed_execution_hash = execution_time_grid_sha256(grid if contextual else grid[0])
        if (
            schedule_binding.execution_time_grid_dtype != IMAGE_EULER_EXECUTION_DTYPE
            or schedule_binding.execution_time_grid_sha256 != observed_execution_hash
            or request.execution_time_grid_sha256 != observed_execution_hash
        ):
            raise ValueError("Executable float32 time grid does not match its request and schedule binding.")
        image_chunks = []
        observed_field_evaluations: set[int] = set()
        with torch.inference_mode():
            for start in range(
                0,
                request.sample_count,
                self.execution_batch_size,
            ):
                stop = min(
                    start + self.execution_batch_size,
                    request.sample_count,
                )
                state_chunk = noise.values[start:stop].to(
                    device=self.device,
                    dtype=torch.float32,
                )
                grid_chunk = grid[start:stop].to(device=self.device)
                label_chunk = (
                    None
                    if request.class_labels is None
                    else torch.tensor(
                        request.class_labels[start:stop],
                        dtype=torch.int64,
                        device=self.device,
                    )
                )

                def field(
                    current: Tensor,
                    progress: Tensor,
                    *,
                    labels: Tensor | None = label_chunk,
                ) -> Tensor:
                    return self.backbone(current, progress, labels)

                result = integrate_euler(
                    field,
                    state_chunk,
                    target_nfe=request.target_nfe,
                    time_grid=grid_chunk,
                )
                observed_field_evaluations.add(result.field_evaluations)
                image_chunks.append(
                    result.final_state.detach().to(
                        device="cpu",
                        dtype=torch.float32,
                    )
                )
        if observed_field_evaluations != {request.target_nfe}:
            raise RuntimeError("Chunked Euler execution violated NFE accounting.")
        images = torch.cat(image_chunks, dim=0).contiguous()
        return GeneratedImageBatch(
            request=request,
            noise=noise,
            schedule=schedule_binding,
            images=images,
            field_evaluations=request.target_nfe,
        )

    def sample_fixed(
        self,
        request: ImageGenerationRequest,
        schedule: FixedSchedule,
    ) -> GeneratedImageBatch:
        self._validate_request(request)
        noise = generate_seeded_image_noise(
            request.dataset_key,
            request.latent_seeds,
        )
        schedule_grid, schedule_binding = _fixed_schedule_grid(
            schedule,
            request=request,
            device=self.device,
        )
        return self._execute(
            request,
            noise=noise,
            time_grid=schedule_grid,
            schedule_binding=schedule_binding,
        )

    def _sample_verified_policy(
        self,
        request: ImageGenerationRequest,
        policy: SchedulePolicy[Tensor],
        *,
        context: Tensor,
        schedule_policy_sha256: str,
    ) -> GeneratedImageBatch:
        self._validate_request(request)
        if not isinstance(context, Tensor):
            raise TypeError("policy context must be a torch.Tensor.")
        if context.ndim < 1 or int(context.shape[0]) != request.sample_count:
            raise ValueError("Policy context must have the request batch size.")
        if not hasattr(policy, "predict") or not callable(policy.predict):
            raise TypeError("policy must implement the neutral SchedulePolicy protocol.")
        noise = generate_seeded_image_noise(
            request.dataset_key,
            request.latent_seeds,
        )
        with torch.inference_mode():
            schedule = policy.predict(
                context,
                target_nfe=request.target_nfe,
            )
        if not isinstance(schedule, ScheduleBatch):
            raise TypeError("SchedulePolicy.predict must return a ScheduleBatch.")
        schedule_binding = _policy_output_binding(
            schedule,
            request=request,
            source_kind="schedule_policy",
            schedule_policy_sha256=schedule_policy_sha256,
        )
        return self._execute(
            request,
            noise=noise,
            time_grid=schedule.time_grid,
            schedule_binding=schedule_binding,
        )

    def sample_policy(
        self,
        request: ImageGenerationRequest,
        policy: IdentifiedSchedulePolicy[Tensor],
        *,
        context: Tensor,
    ) -> GeneratedImageBatch:
        """Execute a content-identified policy for an unconditional backbone.

        Class-conditional GICO must use :meth:`sample_gico`, which derives the
        policy context from the request labels and a verified bound artifact.
        """

        self._validate_request(request)
        if request.class_labels is not None:
            raise ValueError(
                "Class-conditional schedule policies must use sample_gico with a bound artifact."
            )
        if not isinstance(policy, IdentifiedSchedulePolicy):
            raise TypeError(
                "Unconditional policy must implement IdentifiedSchedulePolicy."
            )
        return self._sample_verified_policy(
            request,
            policy,
            context=context,
            schedule_policy_sha256=_sha256_identity(
                policy.policy_sha256,
                field="policy.policy_sha256",
            ),
        )

    def sample_gico(
        self,
        request: ImageGenerationRequest,
        artifact: "BoundImageGICOConditionalArtifact",
    ) -> GeneratedImageBatch:
        """Execute class-conditional ImageNet GICO with label-derived context."""

        from genode.gico.image_conditional_artifacts import (
            BoundImageGICOConditionalArtifact,
        )

        self._validate_request(request)
        if not isinstance(artifact, BoundImageGICOConditionalArtifact):
            raise TypeError("artifact must be a bound ImageNet GICO conditional artifact.")
        if request.class_labels is None:
            raise ValueError("Bound ImageNet GICO execution requires class labels.")
        if artifact.artifact_sha256 != request.schedule_policy_sha256:
            raise ValueError("Bound GICO artifact identity does not match the generation request.")
        binding = artifact.prepared_context.binding
        expected_backbone = (
            request.backbone_manifest.model_key,
            request.backbone_manifest.protocol_sha256,
            request.backbone_manifest.checkpoint.sha256,
        )
        observed_backbone = (
            binding.backbone_model_key,
            binding.backbone_protocol_sha256,
            binding.backbone_checkpoint_sha256,
        )
        if observed_backbone != expected_backbone:
            raise ValueError("Bound GICO context does not match the generation-request backbone.")
        artifact.verify_execution_identity()
        labels = torch.tensor(request.class_labels, dtype=torch.int64)
        context = artifact.contexts_for_class_labels(labels)
        return self._sample_verified_policy(
            request,
            artifact.policy,
            context=context,
            schedule_policy_sha256=artifact.artifact_sha256,
        )

    def sample_gico_clock_barycenter(
        self,
        request: ImageGenerationRequest,
        artifact: "BoundImageGICOClockMixtureArtifact",
    ) -> GeneratedImageBatch:
        """Execute the bound conditional complete-clock barycenter."""

        labels = self._bound_gico_clock_labels(request, artifact)
        with torch.inference_mode():
            schedule = artifact.predict_for_class_labels(
                labels,
                target_nfe=request.target_nfe,
            )
        if not isinstance(schedule, ScheduleBatch):
            raise TypeError(
                "Bound GICO clock-mixture prediction must return a ScheduleBatch."
            )
        schedule_binding = _policy_output_binding(
            schedule,
            request=request,
            source_kind="schedule_policy",
            schedule_policy_sha256=artifact.artifact_sha256,
        )
        noise = generate_seeded_image_noise(
            request.dataset_key,
            request.latent_seeds,
        )
        return self._execute(
            request,
            noise=noise,
            time_grid=schedule.time_grid,
            schedule_binding=schedule_binding,
        )

    def sample_gico_clock_realization(
        self,
        request: ImageGenerationRequest,
        artifact: "BoundImageGICOClockMixtureArtifact",
        *,
        uniforms: Tensor,
        alpha: float,
    ) -> GeneratedImageBatch:
        """Execute an explicit stochastic complete-clock realization."""

        from genode.gico.image_clock_mixture import ImageGICOClockRealization
        labels = self._bound_gico_clock_labels(
            request,
            artifact,
            allow_clock_realization=True,
        )
        if not isinstance(uniforms, Tensor):
            raise TypeError("uniforms must be a torch.Tensor.")
        if uniforms.device.type != "cpu" or uniforms.dtype != torch.float64:
            raise TypeError("uniforms must be a CPU torch.float64 tensor.")
        if uniforms.ndim != 1 or int(uniforms.shape[0]) != request.sample_count:
            raise ValueError("uniforms must have shape [request.sample_count].")
        if (
            not bool(torch.isfinite(uniforms).all())
            or bool(torch.any(uniforms < 0.0))
            or bool(torch.any(uniforms >= 1.0))
        ):
            raise ValueError("uniforms must contain only values in [0, 1).")
        if isinstance(alpha, bool) or not isinstance(alpha, Real):
            raise TypeError("alpha must be a finite real in [0, 1].")
        mixing = float(alpha)
        if not math.isfinite(mixing) or not 0.0 <= mixing <= 1.0:
            raise ValueError("alpha must be a finite real in [0, 1].")

        uniform_snapshot = uniforms.detach().clone(
            memory_format=torch.contiguous_format
        )
        expected_uniforms_sha256 = (
            None
            if mixing == 0.0
            else _clock_uniforms_sha256(uniform_snapshot)
        )
        with torch.inference_mode():
            realization = artifact.realize_for_class_labels(
                labels,
                target_nfe=request.target_nfe,
                uniforms=uniform_snapshot,
                alpha=mixing,
            )
        if not isinstance(realization, ImageGICOClockRealization):
            raise TypeError(
                "Bound GICO clock-mixture realization has an invalid type."
            )
        expected_policy_sha256 = artifact.policy.policy_sha256
        expected_library_sha256 = artifact.library.sha256
        if realization.policy_sha256 != expected_policy_sha256:
            raise ValueError(
                "GICO clock realization does not match the bound policy identity."
            )
        if realization.clock_library_sha256 != expected_library_sha256:
            raise ValueError(
                "GICO clock realization does not match the bound library identity."
            )
        if realization.alpha != mixing:
            raise ValueError("GICO clock realization does not match the requested alpha.")
        realization_binding = ImageGICOClockRealizationBinding.from_realization(
            realization
        )
        if realization_binding.uniforms_sha256 != expected_uniforms_sha256:
            raise ValueError(
                "GICO clock realization does not match the supplied uniforms."
            )
        for index, key in zip(
            realization_binding.selected_schedule_indices,
            realization_binding.selected_schedule_keys,
            strict=True,
        ):
            if index is None:
                continue
            if (
                index >= artifact.library.schedule_count
                or artifact.library.schedule_keys[index] != key
            ):
                raise ValueError(
                    "GICO clock realization selection does not match the bound library."
                )
        if request.clock_realization is None:
            bound_request = replace(
                request,
                clock_realization=realization_binding,
            )
        elif request.clock_realization != realization_binding:
            raise ValueError(
                "Generation request is bound to a different clock realization."
            )
        else:
            bound_request = request
        schedule_binding = _policy_output_binding(
            realization.schedule,
            request=bound_request,
            source_kind="schedule_policy",
            schedule_policy_sha256=artifact.artifact_sha256,
            clock_realization=realization_binding,
        )
        noise = generate_seeded_image_noise(
            bound_request.dataset_key,
            bound_request.latent_seeds,
        )
        return self._execute(
            bound_request,
            noise=noise,
            time_grid=realization.schedule.time_grid,
            schedule_binding=schedule_binding,
        )


__all__ = [
    "IMAGE_EULER_EXECUTION_DTYPE",
    "IMAGE_GENERATED_BATCH_PROTOCOL",
    "IMAGE_GENERATED_BATCH_PROVENANCE_PROTOCOL",
    "IMAGE_GENERATION_REQUEST_PROTOCOL",
    "IMAGE_GICO_CLOCK_REALIZATION_BINDING_PROTOCOL",
    "IMAGE_GICO_CLOCK_REALIZATION_REQUEST_PROTOCOL",
    "GeneratedBatchProvenance",
    "GeneratedImageBatch",
    "ImageGICOClockRealizationBinding",
    "ImageEulerSampler",
    "ImageGenerationRequest",
    "ScheduleExecutionBinding",
    "execution_time_grid_sha256",
    "policy_schedule_request_hashes",
]
