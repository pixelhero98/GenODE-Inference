from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from genode.artifacts.identity import semantic_sha256
from genode.benchmarks.image.protocol import IMAGE_TARGET_NFES
from genode.gico.image_conditional import (
    IMAGE_GICO_BACKBONE_CONTEXT_DIM,
    IMAGE_GICO_CLASS_COUNT,
    IMAGE_GICO_NFE_EMBEDDING_DIM,
    ImageGICOBackboneContextDensityModel,
    ImageGICOBackboneContextModelConfig,
    ImageGICOConditionalTargets,
    validate_image_gico_backbone_context_tensor,
)


IMAGE_GICO_BACKBONE_CONTEXT_TEACHER_PROTOCOL = (
    "image_gico_backbone_context_nfe_density_teacher_v4"
)
IMAGE_GICO_BACKBONE_CONTEXT_TRAINING_PROTOCOL = (
    "image_gico_backbone_context_training_v4"
)
IMAGE_GICO_DENSITY_SUMMARY_PROTOCOL = "image_gico_density_summaries_v1"


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


def _module_state_payload(module: nn.Module) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, tensor in sorted(module.state_dict().items()):
        array = tensor.detach().to(device="cpu").contiguous().numpy()
        payload[name] = {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "values": array.tolist(),
        }
    return payload


def conditional_module_state_sha256(
    module: nn.Module,
    *,
    namespace: str,
) -> str:
    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module.")
    return semantic_sha256(_module_state_payload(module), namespace=namespace)


@dataclass(frozen=True)
class ImageGICOBackboneContextTrainingConfig:
    student_steps: int = 2_000
    teacher_steps: int = 2_000
    teacher_batch_size: int = 512
    student_learning_rate: float = 1e-3
    teacher_learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    residual_penalty_weight: float = 1e-4
    seed: int = 0
    teacher_score_weight: float = 0.01
    teacher_score_warmup_fraction: float = 0.60
    teacher_score_clip: float = 5.0
    teacher_rank_temperature: float = 0.5
    teacher_regression_weight: float = 0.25

    def __post_init__(self) -> None:
        for field in ("student_steps", "teacher_steps", "teacher_batch_size"):
            object.__setattr__(
                self,
                field,
                _positive_integer(getattr(self, field), field=field),
            )
        for field in (
            "student_learning_rate",
            "teacher_learning_rate",
            "weight_decay",
            "residual_penalty_weight",
            "teacher_score_weight",
            "teacher_score_warmup_fraction",
            "teacher_score_clip",
            "teacher_rank_temperature",
            "teacher_regression_weight",
        ):
            object.__setattr__(
                self,
                field,
                _finite_real(getattr(self, field), field=field),
            )
        if self.student_learning_rate <= 0.0 or self.teacher_learning_rate <= 0.0:
            raise ValueError("Learning rates must be positive.")
        if self.teacher_score_clip <= 0.0 or self.teacher_rank_temperature <= 0.0:
            raise ValueError("Teacher score clip and rank temperature must be positive.")
        if not 0.0 <= self.teacher_score_warmup_fraction < 1.0:
            raise ValueError("teacher_score_warmup_fraction must be in [0, 1).")
        if self.teacher_score_weight != 0.01:
            raise ValueError("The conditional experiment fixes teacher_score_weight at 0.01.")
        if self.teacher_score_warmup_fraction != 0.60:
            raise ValueError("The conditional experiment fixes score warmup at 0.60.")
        if self.teacher_score_clip != 5.0:
            raise ValueError("The conditional experiment fixes score clip at 5.0.")
        if self.residual_penalty_weight != 1e-4:
            raise ValueError("The conditional experiment fixes residual penalty at 1e-4.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be a nonnegative integer.")
        if int(self.seed) < 0:
            raise ValueError("seed must be nonnegative.")
        object.__setattr__(self, "seed", int(self.seed))

    def as_payload(self) -> dict[str, Any]:
        return {
            "protocol": IMAGE_GICO_BACKBONE_CONTEXT_TRAINING_PROTOCOL,
            "student_steps": self.student_steps,
            "teacher_steps": self.teacher_steps,
            "teacher_batch_size": self.teacher_batch_size,
            "student_learning_rate": self.student_learning_rate,
            "teacher_learning_rate": self.teacher_learning_rate,
            "weight_decay": self.weight_decay,
            "residual_penalty_weight": self.residual_penalty_weight,
            "seed": self.seed,
            "student_objective": (
                "full_3000_target_kl_plus_centered_residual_penalty"
                "_minus_late_teacher_score"
            ),
            "teacher_objective": (
                "pairwise_rank_plus_standardized_reward_regression"
            ),
            "teacher_cross_validation": (
                "leave_one_complete_schedule_form_out"
            ),
            "teacher_score_weight": self.teacher_score_weight,
            "teacher_score_warmup_fraction": self.teacher_score_warmup_fraction,
            "teacher_score_clip": self.teacher_score_clip,
            "teacher_rank_temperature": self.teacher_rank_temperature,
            "teacher_regression_weight": self.teacher_regression_weight,
            "solver_key": "euler",
            "target_nfes": list(IMAGE_TARGET_NFES),
        }

    @property
    def sha256(self) -> str:
        return semantic_sha256(
            self.as_payload(),
            namespace="image-gico-backbone-context-training-config-v4",
        )


def _density_summaries(density_mass: Tensor) -> Tensor:
    """Differentiable deterministic summaries of a 64-bin density."""

    bins = density_mass.shape[-1]
    centers = (
        torch.arange(
            bins,
            dtype=density_mass.dtype,
            device=density_mass.device,
        )
        + 0.5
    ) / float(bins)
    mean = torch.sum(density_mass * centers, dim=-1, keepdim=True)
    centered = centers - mean
    variance = torch.sum(density_mass * centered.square(), dim=-1, keepdim=True)
    standard_deviation = torch.sqrt(variance.clamp_min(1e-12))
    skewness = torch.sum(density_mass * centered.pow(3), dim=-1, keepdim=True) / (
        standard_deviation.pow(3) + 1e-12
    )
    excess_kurtosis = (
        torch.sum(density_mass * centered.pow(4), dim=-1, keepdim=True)
        / (variance.square() + 1e-12)
        - 3.0
    )
    safe = density_mass.clamp_min(1e-12)
    entropy = -torch.sum(safe * torch.log(safe), dim=-1, keepdim=True) / math.log(
        float(bins)
    )
    total_variation = torch.sum(
        torch.abs(density_mass[..., 1:] - density_mass[..., :-1]),
        dim=-1,
        keepdim=True,
    )
    quarter = bins // 4
    first_quarter = density_mass[..., :quarter].sum(dim=-1, keepdim=True)
    last_quarter = density_mass[..., -quarter:].sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(density_mass, dim=-1)
    quantiles = []
    for quantile in (0.10, 0.25, 0.50, 0.75, 0.90):
        proximity = torch.softmax(-torch.abs(cdf - quantile) / 0.02, dim=-1)
        quantiles.append(torch.sum(proximity * centers, dim=-1, keepdim=True))
    return torch.cat(
        (
            mean,
            standard_deviation,
            skewness,
            excess_kurtosis,
            entropy,
            total_variation,
            first_quarter,
            last_quarter,
            *quantiles,
        ),
        dim=-1,
    )


class ImageGICOBackboneContextTeacher(nn.Module):
    """Score density geometry using a frozen backbone context and target NFE."""

    def __init__(
        self,
        *,
        density_bin_count: int,
        context_dim: int = IMAGE_GICO_BACKBONE_CONTEXT_DIM,
        nfe_embedding_dim: int = IMAGE_GICO_NFE_EMBEDDING_DIM,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.density_bin_count = _positive_integer(
            density_bin_count,
            field="density_bin_count",
        )
        self.context_dim = _positive_integer(context_dim, field="context_dim")
        if self.context_dim != IMAGE_GICO_BACKBONE_CONTEXT_DIM:
            raise ValueError("ImageNet GICO teacher requires 768-dimensional contexts.")
        self.nfe_embedding = nn.Embedding(3, nfe_embedding_dim)
        summary_dim = 13
        input_dim = (
            self.context_dim
            + nfe_embedding_dim
            + self.density_bin_count
            + self.density_bin_count
            - 1
            + summary_dim
        )
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        contexts: Tensor,
        nfe_indices: Tensor,
        density_mass: Tensor,
    ) -> Tensor:
        if not isinstance(contexts, Tensor):
            raise TypeError("Teacher contexts must be a torch.Tensor.")
        if contexts.dtype != torch.float32:
            raise TypeError("Teacher contexts must use torch.float32.")
        if contexts.ndim != 2 or contexts.shape[1] != self.context_dim:
            raise ValueError(
                f"Teacher contexts must have shape [batch, {self.context_dim}]."
            )
        if contexts.shape[0] <= 0 or not bool(torch.isfinite(contexts).all()):
            raise ValueError("Teacher contexts must be nonempty and finite.")
        if nfe_indices.shape != (contexts.shape[0],):
            raise ValueError("Teacher NFE indices must have shape [batch].")
        if nfe_indices.dtype == torch.bool or nfe_indices.is_floating_point():
            raise TypeError("Teacher NFE indices must use an integer dtype.")
        indices = nfe_indices.to(dtype=torch.int64)
        if bool(torch.any(indices < 0)) or bool(torch.any(indices >= 3)):
            raise ValueError("Teacher NFE indices must be in [0, 2].")
        if density_mass.shape != (contexts.shape[0], self.density_bin_count):
            raise ValueError("Teacher density_mass has an incompatible shape.")
        if contexts.device != density_mass.device or indices.device != density_mass.device:
            raise ValueError("Teacher inputs must share a device.")
        if not bool(torch.isfinite(density_mass).all()) or bool(
            torch.any(density_mass < 0.0)
        ):
            raise ValueError("Teacher density_mass must be finite and nonnegative.")
        if not bool(
            torch.allclose(
                density_mass.sum(dim=-1),
                torch.ones(contexts.shape[0], device=density_mass.device),
                rtol=1e-5,
                atol=1e-5,
            )
        ):
            raise ValueError("Teacher density_mass rows must sum to one.")
        safe = density_mass.clamp_min(1e-12)
        cdf = torch.cumsum(density_mass, dim=-1)[..., :-1]
        context = torch.cat(
            (
                contexts,
                self.nfe_embedding(indices),
                torch.log(safe),
                cdf,
                _density_summaries(density_mass),
            ),
            dim=-1,
        )
        return self.network(context).squeeze(-1)


@dataclass(frozen=True)
class ImageGICOBackboneContextTrainingResult:
    model: ImageGICOBackboneContextDensityModel
    teacher: ImageGICOBackboneContextTeacher
    config: ImageGICOBackboneContextTrainingConfig
    context_binding_sha256: str
    target_sha256: str
    feature_group_sha256: str
    final_kl: float
    final_residual_penalty: float
    final_teacher_score: float
    final_objective: float
    conditional_density_range: float
    teacher_schedule_fold_diagnostics: tuple[dict[str, float | int | str], ...]
    teacher_oof_rmse: float
    teacher_oof_pairwise_accuracy: float

    def __post_init__(self) -> None:
        if not isinstance(self.context_binding_sha256, str) or not self.context_binding_sha256:
            raise ValueError("context_binding_sha256 must be a non-empty identity string.")
        for field in (
            "final_kl",
            "final_residual_penalty",
            "final_teacher_score",
            "final_objective",
            "conditional_density_range",
            "teacher_oof_rmse",
            "teacher_oof_pairwise_accuracy",
        ):
            if not math.isfinite(float(getattr(self, field))):
                raise ValueError(f"{field} must be finite.")
        if self.final_residual_penalty < 0.0 or self.conditional_density_range <= 0.0:
            raise ValueError("Training penalties/ranges are inconsistent.")
        if not 0.0 <= self.teacher_oof_pairwise_accuracy <= 1.0:
            raise ValueError("Teacher OOF pairwise accuracy must be in [0, 1].")
        if not self.teacher_schedule_fold_diagnostics:
            raise ValueError("Teacher diagnostics must contain schedule folds.")
        fold_indices = tuple(
            int(item.get("fold", -1))
            for item in self.teacher_schedule_fold_diagnostics
        )
        if fold_indices != tuple(range(len(fold_indices))):
            raise ValueError(
                "Teacher diagnostics must contain one ordered fold per schedule."
            )

    @property
    def model_state_sha256(self) -> str:
        return conditional_module_state_sha256(
            self.model,
            namespace="image-gico-backbone-context-model-state-v4",
        )

    @property
    def teacher_state_sha256(self) -> str:
        return conditional_module_state_sha256(
            self.teacher,
            namespace="image-gico-backbone-context-teacher-state-v4",
        )

    def manifest_payload(self) -> dict[str, Any]:
        schedule_count = len(self.teacher_schedule_fold_diagnostics)
        teacher_evidence_row_count = (
            len(IMAGE_TARGET_NFES) * IMAGE_GICO_CLASS_COUNT * schedule_count
        )
        payload = {
            "protocol": IMAGE_GICO_BACKBONE_CONTEXT_TRAINING_PROTOCOL,
            "conditioning": "normalized_frozen_backbone_map_label_plus_target_nfe",
            "context_binding_sha256": self.context_binding_sha256,
            "feature_group_usage": "reward_shrinkage_only_not_inference_context",
            "target_sha256": self.target_sha256,
            "feature_group_sha256": self.feature_group_sha256,
            "training_config": self.config.as_payload(),
            "training_config_sha256": self.config.sha256,
            "model_config": self.model.config.as_payload(),
            "model_config_sha256": self.model.config.sha256,
            "model_state_sha256": self.model_state_sha256,
            "teacher_protocol": IMAGE_GICO_BACKBONE_CONTEXT_TEACHER_PROTOCOL,
            "teacher_density_summary_protocol": IMAGE_GICO_DENSITY_SUMMARY_PROTOCOL,
            "teacher_state_sha256": self.teacher_state_sha256,
            "teacher_evidence_row_count": teacher_evidence_row_count,
            "teacher_evidence_sha256": semantic_sha256(
                {
                    "target_sha256": self.target_sha256,
                    "row_count": teacher_evidence_row_count,
                    "coverage": (
                        f"all_{IMAGE_GICO_CLASS_COUNT}_classes_x_"
                        f"{schedule_count}_schedules_x_{len(IMAGE_TARGET_NFES)}_nfes"
                    ),
                },
                namespace="image-gico-backbone-context-teacher-evidence-v4",
            ),
            "final_kl": self.final_kl,
            "final_residual_penalty": self.final_residual_penalty,
            "final_teacher_score": self.final_teacher_score,
            "final_objective": self.final_objective,
            "conditional_density_range": self.conditional_density_range,
            "teacher_schedule_fold_diagnostics": [
                dict(item) for item in self.teacher_schedule_fold_diagnostics
            ],
            "teacher_oof_rmse": self.teacher_oof_rmse,
            "teacher_oof_pairwise_accuracy": self.teacher_oof_pairwise_accuracy,
        }
        payload["result_sha256"] = semantic_sha256(
            payload,
            namespace="image-gico-backbone-context-training-result-v4",
        )
        return payload


def _score_weight(step: int, config: ImageGICOBackboneContextTrainingConfig) -> float:
    warmup_steps = int(
        math.floor(config.student_steps * config.teacher_score_warmup_fraction)
    )
    if step < warmup_steps:
        return 0.0
    denominator = max(config.student_steps - warmup_steps - 1, 1)
    progress = float(step - warmup_steps) / float(denominator)
    return config.teacher_score_weight * min(max(progress, 0.0), 1.0)


def _teacher_training_step(
    teacher: ImageGICOBackboneContextTeacher,
    optimizer: torch.optim.Optimizer,
    rewards: Tensor,
    density_sources: Tensor,
    context_table: Tensor,
    *,
    generator: torch.Generator,
    config: ImageGICOBackboneContextTrainingConfig,
    allowed_schedules: Tensor,
) -> None:
    batch = config.teacher_batch_size
    labels = torch.randint(
        0,
        IMAGE_GICO_CLASS_COUNT,
        (batch,),
        generator=generator,
        device=rewards.device,
    )
    nfe_indices = torch.randint(
        0,
        3,
        (batch,),
        generator=generator,
        device=rewards.device,
    )
    left = allowed_schedules[
        torch.randint(
            0,
            allowed_schedules.numel(),
            (batch,),
            generator=generator,
            device=rewards.device,
        )
    ]
    right = allowed_schedules[
        torch.randint(
            0,
            allowed_schedules.numel(),
            (batch,),
            generator=generator,
            device=rewards.device,
        )
    ]
    left_density = density_sources[nfe_indices, left]
    right_density = density_sources[nfe_indices, right]
    left_target = rewards[nfe_indices, labels, left]
    right_target = rewards[nfe_indices, labels, right]
    contexts = context_table[labels]
    left_score = teacher(contexts, nfe_indices, left_density)
    right_score = teacher(contexts, nfe_indices, right_density)
    regression = 0.5 * (
        F.mse_loss(left_score, left_target)
        + F.mse_loss(right_score, right_target)
    )
    target_difference = left_target - right_target
    non_ties = target_difference != 0.0
    if bool(non_ties.any()):
        direction = torch.sign(target_difference[non_ties])
        predicted_difference = left_score[non_ties] - right_score[non_ties]
        rank = F.softplus(
            -direction
            * predicted_difference
            / config.teacher_rank_temperature
        ).mean()
    else:
        rank = regression.new_zeros(())
    loss = rank + config.teacher_regression_weight * regression
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def _train_teacher(
    rewards: Tensor,
    density_sources: Tensor,
    context_table: Tensor,
    *,
    config: ImageGICOBackboneContextTrainingConfig,
    seed: int,
    heldout_schedule: int | None = None,
) -> ImageGICOBackboneContextTeacher:
    schedule_count = int(density_sources.shape[1])
    if schedule_count <= 1:
        raise ValueError("Teacher training requires at least two schedules.")
    teacher = ImageGICOBackboneContextTeacher(
        density_bin_count=int(density_sources.shape[-1])
    ).to(device=rewards.device)
    optimizer = torch.optim.AdamW(
        teacher.parameters(),
        lr=config.teacher_learning_rate,
        weight_decay=config.weight_decay,
    )
    allowed = torch.tensor(
        [
            index
            for index in range(schedule_count)
            if index != heldout_schedule
        ],
        dtype=torch.int64,
        device=rewards.device,
    )
    generator = torch.Generator(device=rewards.device)
    generator.manual_seed(seed)
    for _ in range(config.teacher_steps):
        _teacher_training_step(
            teacher,
            optimizer,
            rewards,
            density_sources,
            context_table,
            generator=generator,
            config=config,
            allowed_schedules=allowed,
        )
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


def _teacher_oof_diagnostics(
    rewards: Tensor,
    density_sources: Tensor,
    context_table: Tensor,
    *,
    schedule_keys: tuple[str, ...],
    config: ImageGICOBackboneContextTrainingConfig,
) -> tuple[tuple[dict[str, float | int | str], ...], float, float]:
    schedule_count = len(schedule_keys)
    if schedule_count <= 1:
        raise ValueError("Teacher diagnostics require at least two schedules.")
    if rewards.shape[-1] != schedule_count or density_sources.shape[1] != schedule_count:
        raise ValueError(
            "Teacher diagnostics schedule tensors must match schedule_keys."
        )
    predictions = torch.empty_like(rewards)
    rows: list[dict[str, float | int | str]] = []
    class_count = context_table.shape[0]
    with torch.enable_grad():
        for heldout in range(schedule_count):
            torch.manual_seed(config.seed + heldout + 1)
            if rewards.device.type == "cuda":
                torch.cuda.manual_seed_all(config.seed + heldout + 1)
            fold_teacher = _train_teacher(
                rewards,
                density_sources,
                context_table,
                config=config,
                seed=config.seed + heldout + 1,
                heldout_schedule=heldout,
            )
            fold_rows = []
            with torch.no_grad():
                for nfe_index in range(3):
                    nfe_indices = torch.full(
                        (class_count,),
                        nfe_index,
                        dtype=torch.int64,
                        device=rewards.device,
                    )
                    density = density_sources[nfe_index, heldout].expand(
                        class_count,
                        -1,
                    )
                    fold_rows.append(
                        fold_teacher(context_table, nfe_indices, density)
                    )
            prediction = torch.stack(fold_rows)
            predictions[..., heldout] = prediction
            target = rewards[..., heldout]
            rows.append(
                {
                    "fold": heldout,
                    "heldout_schedule_key": schedule_keys[heldout],
                    "heldout_row_count": 3 * IMAGE_GICO_CLASS_COUNT,
                    "root_mean_squared_error": float(
                        torch.sqrt(F.mse_loss(prediction, target)).cpu()
                    ),
                    "mean_absolute_error": float(
                        F.l1_loss(prediction, target).cpu()
                    ),
                }
            )
            del fold_teacher
    difference_target = rewards[..., :, None] - rewards[..., None, :]
    difference_prediction = predictions[..., :, None] - predictions[..., None, :]
    upper = torch.triu(
        torch.ones(
            schedule_count,
            schedule_count,
            dtype=torch.bool,
            device=rewards.device,
        ),
        diagonal=1,
    )
    non_ties = (difference_target != 0.0) & upper
    accuracy = (
        torch.sign(difference_target[non_ties])
        == torch.sign(difference_prediction[non_ties])
    ).to(dtype=torch.float64).mean()
    rmse = torch.sqrt(F.mse_loss(predictions, rewards))
    return tuple(rows), float(rmse.cpu()), float(accuracy.cpu())


def train_image_gico_backbone_context(
    targets: ImageGICOConditionalTargets,
    *,
    fixed_density_mass: np.ndarray | Tensor,
    normalized_context_table: np.ndarray | Tensor,
    context_binding_sha256: str,
    config: ImageGICOBackboneContextTrainingConfig | None = None,
    device: torch.device | str = "cpu",
) -> ImageGICOBackboneContextTrainingResult:
    """Train diagnostics, teacher, and student from frozen backbone contexts."""

    if not isinstance(targets, ImageGICOConditionalTargets):
        raise TypeError("targets must be ImageGICOConditionalTargets.")
    if not isinstance(context_binding_sha256, str) or not context_binding_sha256:
        raise ValueError("context_binding_sha256 must be a non-empty identity string.")
    training = ImageGICOBackboneContextTrainingConfig() if config is None else config
    if not isinstance(training, ImageGICOBackboneContextTrainingConfig):
        raise TypeError("config must be ImageGICOBackboneContextTrainingConfig.")
    execution_device = torch.device(device)
    context_table = validate_image_gico_backbone_context_tensor(
        normalized_context_table,
        field="normalized_context_table",
        expected_rows=IMAGE_GICO_CLASS_COUNT,
        device=execution_device,
    )
    torch.manual_seed(training.seed)
    if execution_device.type == "cuda":
        torch.cuda.manual_seed_all(training.seed)
    density_sources = torch.as_tensor(
        fixed_density_mass,
        dtype=torch.float32,
        device=execution_device,
    )
    schedule_count = len(targets.schedule_keys)
    expected_density_shape = (
        len(IMAGE_TARGET_NFES),
        schedule_count,
        targets.density_bin_count,
    )
    if density_sources.shape != expected_density_shape:
        raise ValueError(
            f"fixed_density_mass must have shape {expected_density_shape}."
        )
    if bool(torch.any(density_sources < 0.0)) or not bool(
        torch.allclose(
            density_sources.sum(dim=-1),
            torch.ones(
                len(IMAGE_TARGET_NFES),
                schedule_count,
                device=execution_device,
            ),
            rtol=1e-5,
            atol=1e-5,
        )
    ):
        raise ValueError("fixed_density_mass rows must be normalized and nonnegative.")
    rewards = torch.tensor(
        targets.normalized_rewards,
        dtype=torch.float32,
        device=execution_device,
    )
    target_density = targets.density_tensor(
        dtype=torch.float32,
        device=execution_device,
    )
    diagnostics, oof_rmse, oof_accuracy = _teacher_oof_diagnostics(
        rewards,
        density_sources,
        context_table,
        schedule_keys=targets.schedule_keys,
        config=training,
    )
    torch.manual_seed(training.seed)
    if execution_device.type == "cuda":
        torch.cuda.manual_seed_all(training.seed)
    teacher = _train_teacher(
        rewards,
        density_sources,
        context_table,
        config=training,
        seed=training.seed,
    )
    model = ImageGICOBackboneContextDensityModel(
        ImageGICOBackboneContextModelConfig(
            density_bin_count=targets.density_bin_count,
        ),
        context_table,
    ).to(device=execution_device)
    with torch.no_grad():
        global_target = target_density.mean(dim=1)
        model.global_logits_by_nfe.copy_(torch.log(global_target))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.student_learning_rate,
        weight_decay=training.weight_decay,
    )
    final_kl = math.nan
    final_residual = math.nan
    final_score = math.nan
    final_objective = math.nan
    for step in range(training.student_steps):
        predicted = model.canonical_density_table()
        kl = torch.sum(
            target_density
            * (torch.log(target_density) - torch.log(predicted)),
            dim=-1,
        ).mean()
        residual_penalty = model.centered_residual_table().square().mean()
        flat_contexts = context_table.repeat(3, 1)
        flat_nfes = torch.arange(
            3,
            dtype=torch.int64,
            device=execution_device,
        ).repeat_interleave(IMAGE_GICO_CLASS_COUNT)
        teacher_score = teacher(
            flat_contexts,
            flat_nfes,
            predicted.reshape(-1, targets.density_bin_count),
        ).clamp(
            min=-training.teacher_score_clip,
            max=training.teacher_score_clip,
        ).mean()
        weight = _score_weight(step, training)
        objective = (
            kl
            + training.residual_penalty_weight * residual_penalty
            - weight * teacher_score
        )
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        final_kl = float(kl.detach().cpu())
        final_residual = float(residual_penalty.detach().cpu())
        final_score = float(teacher_score.detach().cpu())
        final_objective = float(objective.detach().cpu())
    model.eval()
    with torch.no_grad():
        table = model.canonical_density_table()
    conditional_range = float(
        (table.amax(dim=1) - table.amin(dim=1)).amax().detach().cpu()
    )
    return ImageGICOBackboneContextTrainingResult(
        model=model,
        teacher=teacher,
        config=training,
        context_binding_sha256=context_binding_sha256,
        target_sha256=targets.sha256,
        feature_group_sha256=targets.feature_group_sha256,
        final_kl=final_kl,
        final_residual_penalty=final_residual,
        final_teacher_score=final_score,
        final_objective=final_objective,
        conditional_density_range=conditional_range,
        teacher_schedule_fold_diagnostics=diagnostics,
        teacher_oof_rmse=oof_rmse,
        teacher_oof_pairwise_accuracy=oof_accuracy,
    )


__all__ = [
    "IMAGE_GICO_BACKBONE_CONTEXT_TEACHER_PROTOCOL",
    "IMAGE_GICO_BACKBONE_CONTEXT_TRAINING_PROTOCOL",
    "IMAGE_GICO_DENSITY_SUMMARY_PROTOCOL",
    "ImageGICOBackboneContextTeacher",
    "ImageGICOBackboneContextTrainingConfig",
    "ImageGICOBackboneContextTrainingResult",
    "conditional_module_state_sha256",
    "train_image_gico_backbone_context",
]
