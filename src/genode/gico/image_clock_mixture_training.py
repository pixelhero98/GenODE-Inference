from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import math
from numbers import Integral, Real
import os
import re
import threading
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch import Tensor

from genode.artifacts.identity import canonical_json_text, semantic_sha256
from genode.benchmarks.image.protocol import IMAGE_TARGET_NFES
from genode.gico.image_clock_mixture import (
    ImageGICOBackboneContextClockMixtureModel,
    ImageGICOClockLibrary,
    ImageGICOClockMixtureModelConfig,
)
from genode.gico.image_conditional import (
    IMAGE_GICO_BACKBONE_CONTEXT_DIM,
    IMAGE_GICO_CLASS_COUNT,
    ImageGICOConditionalTargets,
    validate_image_gico_backbone_context_tensor,
)


IMAGE_GICO_CLOCK_MIXTURE_TRAINING_PROTOCOL = (
    "image_gico_complete_clock_mixture_training_v1"
)
_CONTEXT_TABLE_PROTOCOL = "image-gico-normalized-context-training-table-v1"
_DETERMINISM_PROTOCOL = "image_gico_clock_mixture_determinism_v1"
_IDENTITY = re.compile(r"^[a-z0-9][a-z0-9-]*:[0-9a-f]{64}$")
_CUDA_CUBLAS_WORKSPACE_CONFIGS = frozenset({":16:8", ":4096:8"})
# PyTorch determinism switches and default generators are process-global.
_DETERMINISTIC_TRAINING_LOCK = threading.Lock()


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be a positive integer.")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field} must be positive.")
    return parsed


def _finite_real(
    value: object,
    *,
    field: str,
    minimum: float = 0.0,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a finite real.")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < minimum:
        raise ValueError(f"{field} must be finite and at least {minimum}.")
    return parsed


def _identity(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTITY.fullmatch(value) is None:
        raise ValueError(f"{field} must be a namespaced SHA-256 identity.")
    return value


def _tensor_payload(tensor: Tensor) -> dict[str, Any]:
    if tensor.layout != torch.strided:
        raise ValueError("Clock-mixture state tensors must use strided layout.")
    array = tensor.detach().to(device="cpu").contiguous().numpy()
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "content_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _context_table_sha256(context_table: Tensor) -> str:
    return semantic_sha256(
        {
            "context_dim": IMAGE_GICO_BACKBONE_CONTEXT_DIM,
            "class_count": IMAGE_GICO_CLASS_COUNT,
            "table": _tensor_payload(context_table),
        },
        namespace=_CONTEXT_TABLE_PROTOCOL,
    )


def _determinism_contract(
    *,
    execution_device_type: str,
    cuda_cublas_workspace_config: str | None,
) -> dict[str, object]:
    return {
        "protocol": _DETERMINISM_PROTOCOL,
        "rng_isolation": "torch.random.fork_rng",
        "deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": False,
        "float32_matmul_precision": "highest",
        "cuda_cublas_workspace_config": cuda_cublas_workspace_config,
        "cuda_tf32_enabled": False if execution_device_type == "cuda" else None,
        "cudnn_benchmark": False if execution_device_type == "cuda" else None,
        "cudnn_deterministic": True if execution_device_type == "cuda" else None,
        "optimizer_foreach": False,
    }


@contextmanager
def _deterministic_training_scope(
    device: torch.device,
) -> Iterator[str | None]:
    with _DETERMINISTIC_TRAINING_LOCK:
        cuda_workspace = None
        if device.type == "cuda":
            cuda_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            if cuda_workspace not in _CUDA_CUBLAS_WORKSPACE_CONFIGS:
                raise RuntimeError(
                    "Deterministic CUDA clock-mixture training requires "
                    "CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) before Python starts."
                )

        previous_algorithms = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        previous_precision = torch.get_float32_matmul_precision()
        previous_cudnn_benchmark = torch.backends.cudnn.benchmark
        previous_cudnn_deterministic = torch.backends.cudnn.deterministic
        previous_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
        previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.set_float32_matmul_precision("highest")
            if device.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            yield cuda_workspace
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_cuda_tf32
            torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
            torch.backends.cudnn.benchmark = previous_cudnn_benchmark
            torch.backends.cudnn.deterministic = previous_cudnn_deterministic
            torch.set_float32_matmul_precision(previous_precision)
            torch.use_deterministic_algorithms(
                previous_algorithms,
                warn_only=previous_warn_only,
            )


@dataclass(frozen=True, slots=True)
class ImageGICOClockMixtureTrainingConfig:
    steps: int = 2_000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    residual_penalty_weight: float = 1e-4
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "steps",
            _positive_integer(self.steps, field="steps"),
        )
        for field_name in (
            "learning_rate",
            "weight_decay",
            "residual_penalty_weight",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_real(
                    getattr(self, field_name),
                    field=field_name,
                ),
            )
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.residual_penalty_weight != 1e-4:
            raise ValueError("Complete-clock training fixes residual_penalty_weight at 1e-4.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be a nonnegative integer.")
        seed = int(self.seed)
        if not 0 <= seed <= (2**63 - 1):
            raise ValueError("seed must be in [0, 2**63 - 1].")
        object.__setattr__(self, "seed", seed)

    def as_payload(self) -> dict[str, Any]:
        return {
            "protocol": IMAGE_GICO_CLOCK_MIXTURE_TRAINING_PROTOCOL,
            "conditioning": "normalized_frozen_backbone_map_label_plus_target_nfe",
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "residual_penalty_weight": self.residual_penalty_weight,
            "seed": self.seed,
            "optimizer": "adamw_foreach_false",
            "objective": "grouped_target_cross_entropy_plus_centered_residual_penalty",
            "primary_divergence": "kl_target_group_probability_to_model_group_probability",
            "teacher_scoring": "none",
            "training_rows": len(IMAGE_TARGET_NFES) * IMAGE_GICO_CLASS_COUNT,
            "target_nfes": list(IMAGE_TARGET_NFES),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> ImageGICOClockMixtureTrainingConfig:
        """Reconstruct a config only from its exact canonical manifest schema."""

        expected_fields = {
            "protocol",
            "conditioning",
            "steps",
            "learning_rate",
            "weight_decay",
            "residual_penalty_weight",
            "seed",
            "optimizer",
            "objective",
            "primary_divergence",
            "teacher_scoring",
            "training_rows",
            "target_nfes",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected_fields:
            raise ValueError(
                "Clock-mixture training-config fields are incomplete or unexpected."
            )
        config = cls(
            steps=payload["steps"],
            learning_rate=payload["learning_rate"],
            weight_decay=payload["weight_decay"],
            residual_penalty_weight=payload["residual_penalty_weight"],
            seed=payload["seed"],
        )
        if canonical_json_text(dict(payload)) != canonical_json_text(
            config.as_payload()
        ):
            raise ValueError("Clock-mixture training config is inconsistent.")
        return config

    @property
    def sha256(self) -> str:
        return semantic_sha256(
            self.as_payload(),
            namespace="image-gico-complete-clock-mixture-training-config-v1",
        )


@dataclass(frozen=True, slots=True)
class ImageGICOClockMixtureTrainingResult:
    model: ImageGICOBackboneContextClockMixtureModel
    config: ImageGICOClockMixtureTrainingConfig
    context_binding_sha256: str
    normalized_context_table_sha256: str
    target_sha256: str
    clock_library_sha256: str
    target_nfes: tuple[int, ...]
    schedule_keys: tuple[str, ...]
    group_count_by_nfe: tuple[int, ...]
    execution_device_type: str
    cuda_cublas_workspace_config: str | None
    target_reconstruction_max_abs_error: float
    final_cross_entropy: float
    final_kl: float
    final_residual_penalty: float
    final_objective: float
    final_barycenter_l1: float
    max_probability_error: float
    _model_state_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model, ImageGICOBackboneContextClockMixtureModel):
            raise TypeError("model must be an ImageGICOBackboneContextClockMixtureModel.")
        if not isinstance(self.config, ImageGICOClockMixtureTrainingConfig):
            raise TypeError("config must be ImageGICOClockMixtureTrainingConfig.")
        if self.model.training:
            raise ValueError("The returned clock-mixture model must be in eval mode.")
        object.__setattr__(
            self,
            "_model_state_sha256",
            self.model.state_sha256,
        )
        for field_name in (
            "context_binding_sha256",
            "normalized_context_table_sha256",
            "target_sha256",
            "clock_library_sha256",
        ):
            _identity(getattr(self, field_name), field=field_name)
        if tuple(self.target_nfes) != tuple(IMAGE_TARGET_NFES):
            raise ValueError(f"target_nfes must be exactly {IMAGE_TARGET_NFES}.")
        if not self.schedule_keys or len(self.schedule_keys) != len(set(self.schedule_keys)):
            raise ValueError("schedule_keys must be nonempty and unique.")
        if len(self.group_count_by_nfe) != len(self.target_nfes) or any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or not 1 <= int(value) <= len(self.schedule_keys)
            for value in self.group_count_by_nfe
        ):
            raise ValueError("group_count_by_nfe is inconsistent with schedule coverage.")
        if self.execution_device_type not in {"cpu", "cuda"}:
            raise ValueError("execution_device_type must be 'cpu' or 'cuda'.")
        if self.execution_device_type == "cpu":
            if self.cuda_cublas_workspace_config is not None:
                raise ValueError("CPU training cannot declare a CUDA workspace config.")
        elif self.cuda_cublas_workspace_config not in _CUDA_CUBLAS_WORKSPACE_CONFIGS:
            raise ValueError("CUDA training must declare its deterministic cuBLAS workspace config.")
        for field_name in (
            "target_reconstruction_max_abs_error",
            "final_cross_entropy",
            "final_kl",
            "final_residual_penalty",
            "final_objective",
            "final_barycenter_l1",
            "max_probability_error",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{field_name} must be finite and nonnegative."
                )
        if self.max_probability_error > 1.0:
            raise ValueError("max_probability_error must be at most one.")
        expected_objective = (
            self.final_cross_entropy
            + self.config.residual_penalty_weight * self.final_residual_penalty
        )
        if not math.isclose(
            self.final_objective,
            expected_objective,
            rel_tol=1e-7,
            abs_tol=1e-9,
        ):
            raise ValueError("final_objective is inconsistent with its declared terms.")

    @property
    def model_state_sha256(self) -> str:
        return self._model_state_sha256

    def identity_payload(self) -> dict[str, Any]:
        return {
            "protocol": IMAGE_GICO_CLOCK_MIXTURE_TRAINING_PROTOCOL,
            "conditioning": "normalized_frozen_backbone_map_label_plus_target_nfe",
            "context_binding_sha256": self.context_binding_sha256,
            "normalized_context_table_sha256": self.normalized_context_table_sha256,
            "target_sha256": self.target_sha256,
            "clock_library_sha256": self.clock_library_sha256,
            "training_config": self.config.as_payload(),
            "training_config_sha256": self.config.sha256,
            "model_config": self.model.config.as_payload(),
            "model_config_sha256": self.model.config.sha256,
            "model_state_sha256": self.model_state_sha256,
            "coverage": {
                "class_count": IMAGE_GICO_CLASS_COUNT,
                "target_nfes": list(self.target_nfes),
                "schedule_keys": list(self.schedule_keys),
                "schedule_count": len(self.schedule_keys),
                "group_count_by_nfe": list(self.group_count_by_nfe),
                "class_nfe_row_count": len(self.target_nfes) * IMAGE_GICO_CLASS_COUNT,
                "all_target_mixture_weights_consumed": True,
            },
            "execution_device_type": self.execution_device_type,
            "determinism": _determinism_contract(
                execution_device_type=self.execution_device_type,
                cuda_cublas_workspace_config=self.cuda_cublas_workspace_config,
            ),
            "training_dtype": "float32",
            "diagnostic_dtype": "float64",
            "target_reconstruction_max_abs_error": self.target_reconstruction_max_abs_error,
            "final_cross_entropy": self.final_cross_entropy,
            "final_kl": self.final_kl,
            "final_residual_penalty": self.final_residual_penalty,
            "final_objective": self.final_objective,
            "final_barycenter_l1": self.final_barycenter_l1,
            "max_probability_error": self.max_probability_error,
            "teacher_scoring": "none",
            "interpolated_density_teacher_scoring": False,
        }

    @property
    def sha256(self) -> str:
        return semantic_sha256(
            self.identity_payload(),
            namespace="image-gico-complete-clock-mixture-training-result-v1",
        )

    def manifest_payload(self) -> dict[str, Any]:
        return {**self.identity_payload(), "result_sha256": self.sha256}


def _aggregate_groups(
    schedule_probabilities: Tensor,
    groups: tuple[tuple[int, ...], ...],
) -> Tensor:
    return torch.stack(
        tuple(
            schedule_probabilities[..., list(member_indices)].sum(dim=-1)
            for member_indices in groups
        ),
        dim=-1,
    )


def _group_target_tables(
    target_schedule_probabilities: Tensor,
    groups: tuple[tuple[tuple[int, ...], ...], ...],
) -> tuple[Tensor, ...]:
    return tuple(
        _aggregate_groups(target_schedule_probabilities[nfe_index], nfe_groups)
        for nfe_index, nfe_groups in enumerate(groups)
    )


def _grouped_cross_entropy(
    schedule_probabilities: Tensor,
    group_targets: tuple[Tensor, ...],
    groups: tuple[tuple[tuple[int, ...], ...], ...],
) -> Tensor:
    terms = []
    for nfe_index, nfe_groups in enumerate(groups):
        predicted = _aggregate_groups(schedule_probabilities[nfe_index], nfe_groups)
        if bool(torch.any(predicted <= 0.0)) or not bool(torch.isfinite(predicted).all()):
            raise FloatingPointError("Clock-mixture group probabilities must be finite and positive.")
        target = group_targets[nfe_index]
        terms.append(-(target * torch.log(predicted)).sum(dim=-1).mean())
    return torch.stack(terms).mean()


def _validate_library_target_binding(
    library: ImageGICOClockLibrary,
    targets: ImageGICOConditionalTargets,
) -> float:
    if tuple(library.target_nfes) != tuple(targets.target_nfes):
        raise ValueError("Clock library and conditional targets disagree on target NFEs.")
    if tuple(library.schedule_keys) != tuple(targets.schedule_keys):
        raise ValueError("Clock library and conditional targets disagree on schedule keys or order.")
    if tuple(library.schedule_sha256s) != tuple(targets.schedule_sha256s):
        raise ValueError("Clock library and conditional targets disagree on schedule identities.")
    if library.target_sha256 != targets.sha256:
        raise ValueError("Clock library is not bound to the supplied conditional targets.")
    library.validate_targets(targets)
    weights = torch.tensor(targets.mixture_weights, dtype=torch.float64)
    supervision = library.supervision_density_mass.to(
        device="cpu",
        dtype=torch.float64,
    )
    expected_shape = (
        len(targets.target_nfes),
        len(targets.schedule_keys),
        targets.density_bin_count,
    )
    if tuple(supervision.shape) != expected_shape:
        raise ValueError(
            f"Clock-library supervision_density_mass must have shape {expected_shape}."
        )
    reconstructed = torch.einsum("ncs,nsb->ncb", weights, supervision)
    declared = torch.tensor(targets.density_mass, dtype=torch.float64)
    error = float(torch.max(torch.abs(reconstructed - declared)))
    if not torch.allclose(reconstructed, declared, rtol=1e-7, atol=1e-7):
        raise ValueError(
            "Conditional target density_mass is not reconstructed by its declared "
            "mixture weights and clock-library supervision support."
        )
    return error


def _execution_device(value: torch.device | str) -> torch.device:
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Clock-mixture training supports only CPU and CUDA devices.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA clock-mixture training was requested but CUDA is unavailable.")
        index = torch.cuda.current_device() if device.index is None else device.index
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is unavailable.")
        return torch.device("cuda", index)
    if device.index is not None:
        raise ValueError("CPU training devices may not specify an index.")
    return device


def _initialize_global_logits(
    model: ImageGICOBackboneContextClockMixtureModel,
    group_targets: tuple[Tensor, ...],
    groups: tuple[tuple[tuple[int, ...], ...], ...],
) -> None:
    expected_shape = (len(IMAGE_TARGET_NFES), len(model.library.schedule_keys))
    if tuple(model.global_logits_by_nfe.shape) != expected_shape:
        raise ValueError(
            f"Clock-mixture global logits must have shape {expected_shape}."
        )
    values = torch.empty_like(model.global_logits_by_nfe)
    for nfe_index, nfe_groups in enumerate(groups):
        mean_target = group_targets[nfe_index].mean(dim=0)
        if bool(torch.any(mean_target <= 0.0)) or not bool(torch.isfinite(mean_target).all()):
            raise ValueError("Every mean target group probability must be finite and positive.")
        for group_index, member_indices in enumerate(nfe_groups):
            values[nfe_index, list(member_indices)] = torch.log(mean_target[group_index])
    with torch.no_grad():
        model.global_logits_by_nfe.copy_(values)


def _final_diagnostics(
    model: ImageGICOBackboneContextClockMixtureModel,
    library: ImageGICOClockLibrary,
    target_schedule_probabilities: Tensor,
    group_targets: tuple[Tensor, ...],
    groups: tuple[tuple[tuple[int, ...], ...], ...],
    *,
    residual_penalty_weight: float,
) -> tuple[float, float, float, float, float, float]:
    with torch.no_grad():
        schedule_probabilities = model.canonical_schedule_probability_table()
        residual = model.centered_residual_table().to(dtype=torch.float64).square().mean()
        cross_entropy_terms = []
        kl_terms = []
        maximum_error = schedule_probabilities.new_zeros((), dtype=torch.float64)
        for nfe_index, nfe_groups in enumerate(groups):
            predicted = _aggregate_groups(
                schedule_probabilities[nfe_index].to(dtype=torch.float64),
                nfe_groups,
            )
            target = group_targets[nfe_index].to(dtype=torch.float64)
            if bool(torch.any(predicted <= 0.0)) or not bool(torch.isfinite(predicted).all()):
                raise FloatingPointError("Final clock-mixture probabilities must be finite and positive.")
            cross_entropy_terms.append(
                -(target * torch.log(predicted)).sum(dim=-1).mean()
            )
            positive = target > 0.0
            kl_terms.append(
                torch.where(
                    positive,
                    target * (torch.log(target.clamp_min(torch.finfo(torch.float64).tiny)) - torch.log(predicted)),
                    torch.zeros_like(target),
                ).sum(dim=-1).mean()
            )
            maximum_error = torch.maximum(
                maximum_error,
                torch.max(torch.abs(predicted - target)),
            )
        cross_entropy = torch.stack(cross_entropy_terms).mean()
        kl = torch.stack(kl_terms).mean()
        if float(kl) < -1e-10:
            raise FloatingPointError("Final grouped KL is numerically negative.")
        kl = kl.clamp_min(0.0)
        exact_bank = library.density_mass.to(
            device=schedule_probabilities.device,
            dtype=torch.float64,
        )
        predicted_barycenter = torch.einsum(
            "ncs,nsb->ncb",
            schedule_probabilities.to(dtype=torch.float64),
            exact_bank,
        )
        target_barycenter = torch.einsum(
            "ncs,nsb->ncb",
            target_schedule_probabilities.to(dtype=torch.float64),
            exact_bank,
        )
        barycenter_l1 = (
            torch.abs(predicted_barycenter - target_barycenter)
            .sum(dim=-1)
            .mean()
        )
        objective = cross_entropy + residual_penalty_weight * residual
    values = (
        cross_entropy,
        kl,
        residual,
        objective,
        barycenter_l1,
        maximum_error,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise FloatingPointError("Clock-mixture final diagnostics must be finite.")
    return (
        float(cross_entropy.detach().cpu()),
        float(kl.detach().cpu()),
        float(residual.detach().cpu()),
        float(objective.detach().cpu()),
        float(barycenter_l1.detach().cpu()),
        float(maximum_error.detach().cpu()),
    )


def recompute_image_gico_clock_mixture_diagnostics(
    model: ImageGICOBackboneContextClockMixtureModel,
    targets: ImageGICOConditionalTargets,
    config: ImageGICOClockMixtureTrainingConfig,
) -> dict[str, float]:
    """Recompute all result diagnostics from a bound frozen model and targets."""

    if type(model) is not ImageGICOBackboneContextClockMixtureModel:
        raise TypeError(
            "model must be an ImageGICOBackboneContextClockMixtureModel."
        )
    if not isinstance(targets, ImageGICOConditionalTargets):
        raise TypeError("targets must be ImageGICOConditionalTargets.")
    if not isinstance(config, ImageGICOClockMixtureTrainingConfig):
        raise TypeError("config must be ImageGICOClockMixtureTrainingConfig.")
    if model.training:
        raise ValueError("Diagnostic recomputation requires an eval-mode model.")
    library = model.library
    reconstruction_error = _validate_library_target_binding(library, targets)
    groups = tuple(
        tuple(group.member_indices for group in nfe_groups)
        for nfe_groups in library.groups
    )
    device = model.global_logits_by_nfe.device
    target_schedule_probabilities = torch.tensor(
        targets.mixture_weights,
        dtype=torch.float64,
        device=device,
    )
    group_targets = _group_target_tables(
        target_schedule_probabilities,
        groups,
    )
    (
        final_cross_entropy,
        final_kl,
        final_residual_penalty,
        final_objective,
        final_barycenter_l1,
        max_probability_error,
    ) = _final_diagnostics(
        model,
        library,
        target_schedule_probabilities,
        group_targets,
        groups,
        residual_penalty_weight=config.residual_penalty_weight,
    )
    return {
        "target_reconstruction_max_abs_error": reconstruction_error,
        "final_cross_entropy": final_cross_entropy,
        "final_kl": final_kl,
        "final_residual_penalty": final_residual_penalty,
        "final_objective": final_objective,
        "final_barycenter_l1": final_barycenter_l1,
        "max_probability_error": max_probability_error,
    }


def train_image_gico_clock_mixture(
    targets: ImageGICOConditionalTargets,
    *,
    library: ImageGICOClockLibrary,
    normalized_context_table: np.ndarray | Tensor,
    context_binding_sha256: str,
    config: ImageGICOClockMixtureTrainingConfig | None = None,
    device: torch.device | str = "cpu",
) -> ImageGICOClockMixtureTrainingResult:
    """Fit context-conditional probabilities over complete supervised clocks."""

    if not isinstance(targets, ImageGICOConditionalTargets):
        raise TypeError("targets must be ImageGICOConditionalTargets.")
    if not isinstance(library, ImageGICOClockLibrary):
        raise TypeError("library must be ImageGICOClockLibrary.")
    binding_sha256 = _identity(
        context_binding_sha256,
        field="context_binding_sha256",
    )
    training = ImageGICOClockMixtureTrainingConfig() if config is None else config
    if not isinstance(training, ImageGICOClockMixtureTrainingConfig):
        raise TypeError("config must be ImageGICOClockMixtureTrainingConfig.")
    execution_device = _execution_device(device)
    reconstruction_error = _validate_library_target_binding(library, targets)
    context_table = validate_image_gico_backbone_context_tensor(
        normalized_context_table,
        field="normalized_context_table",
        expected_rows=IMAGE_GICO_CLASS_COUNT,
        device="cpu",
    )
    context_table_sha256 = _context_table_sha256(context_table)
    groups = tuple(
        tuple(group.member_indices for group in nfe_groups)
        for nfe_groups in library.groups
    )
    group_counts = library.group_counts
    cuda_devices = (
        []
        if execution_device.type == "cpu"
        else [execution_device.index]
    )
    with (
        _deterministic_training_scope(execution_device) as cuda_workspace,
        torch.random.fork_rng(devices=cuda_devices),
    ):
        torch.random.default_generator.manual_seed(training.seed)
        if execution_device.type == "cuda":
            with torch.cuda.device(execution_device):
                torch.cuda.manual_seed(training.seed)
        model = ImageGICOBackboneContextClockMixtureModel(
            ImageGICOClockMixtureModelConfig.for_library(library),
            context_table,
            library,
        ).to(device=execution_device)
        diagnostic_target_schedule_probabilities = torch.tensor(
            targets.mixture_weights,
            dtype=torch.float64,
            device=execution_device,
        )
        target_schedule_probabilities = diagnostic_target_schedule_probabilities.to(
            dtype=torch.float32,
        )
        group_targets = _group_target_tables(
            target_schedule_probabilities,
            groups,
        )
        diagnostic_group_targets = _group_target_tables(
            diagnostic_target_schedule_probabilities,
            groups,
        )
        _initialize_global_logits(model, group_targets, groups)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training.learning_rate,
            weight_decay=training.weight_decay,
            foreach=False,
        )
        model.train()
        for _ in range(training.steps):
            schedule_probabilities = model.canonical_schedule_probability_table()
            cross_entropy = _grouped_cross_entropy(
                schedule_probabilities,
                group_targets,
                groups,
            )
            residual_penalty = model.centered_residual_table().square().mean()
            objective = (
                cross_entropy
                + training.residual_penalty_weight * residual_penalty
            )
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError("Clock-mixture training objective became non-finite.")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
        model.eval()
        (
            final_cross_entropy,
            final_kl,
            final_residual_penalty,
            final_objective,
            final_barycenter_l1,
            max_probability_error,
        ) = _final_diagnostics(
            model,
            library,
            diagnostic_target_schedule_probabilities,
            diagnostic_group_targets,
            groups,
            residual_penalty_weight=training.residual_penalty_weight,
        )
    return ImageGICOClockMixtureTrainingResult(
        model=model,
        config=training,
        context_binding_sha256=binding_sha256,
        normalized_context_table_sha256=context_table_sha256,
        target_sha256=targets.sha256,
        clock_library_sha256=library.sha256,
        target_nfes=tuple(library.target_nfes),
        schedule_keys=tuple(library.schedule_keys),
        group_count_by_nfe=group_counts,
        execution_device_type=execution_device.type,
        cuda_cublas_workspace_config=cuda_workspace,
        target_reconstruction_max_abs_error=reconstruction_error,
        final_cross_entropy=final_cross_entropy,
        final_kl=final_kl,
        final_residual_penalty=final_residual_penalty,
        final_objective=final_objective,
        final_barycenter_l1=final_barycenter_l1,
        max_probability_error=max_probability_error,
    )


__all__ = [
    "IMAGE_GICO_CLOCK_MIXTURE_TRAINING_PROTOCOL",
    "ImageGICOClockMixtureTrainingConfig",
    "ImageGICOClockMixtureTrainingResult",
    "recompute_image_gico_clock_mixture_diagnostics",
    "train_image_gico_clock_mixture",
]
