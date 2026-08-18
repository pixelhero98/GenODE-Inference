"""Training objective for the stochastic causal-AR image GICO student."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from genode.gico.image_causal_policy import (
    EXPECTED_TRAINABLE_PARAMETER_COUNT,
    ImageGICOCausalConfig,
    ImageGICOCausalTransformer,
)
from genode.gico.image_causal_stick import (
    STICK_ACTION_COUNT,
    TARGET_NFES,
    TOKEN_COUNT,
    ImageGICOCausalPathBank,
)
from genode.gico.image_supervision import ImageGICOSupervision
from genode.gico.image_training_rng import (
    resolve_image_gico_training_device,
    seed_image_gico_training_generators,
)

CAUSAL_TRAINING_CONFIG_PROTOCOL = "gico-causal-transformer-training-config-v2"
CAUSAL_TRAINING_REPORT_PROTOCOL = "gico-causal-transformer-training-report-v3"
IMAGE_GICO_CAUSAL_STATE_NAMESPACE = "image-gico-causal-ar-state-v2"


def image_gico_causal_state_sha256(
    state_or_model: Mapping[str, Tensor] | ImageGICOCausalTransformer,
) -> str:
    """Hash the qualified causal state layout without changing its protocol."""

    state = state_or_model.state_dict() if isinstance(state_or_model, ImageGICOCausalTransformer) else state_or_model
    if not isinstance(state, Mapping) or not state:
        raise TypeError("state must be a nonempty tensor mapping.")
    digest = hashlib.sha256()
    digest.update(IMAGE_GICO_CAUSAL_STATE_NAMESPACE.encode("ascii"))
    digest.update(b"\0")
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            raise TypeError("state must map parameter names to tensors.")
        value = tensor.detach().to(device="cpu").contiguous()
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"State tensor {name!r} is non-finite.")
        encoded_name = name.encode("utf-8")
        encoded_dtype = str(value.dtype).encode("ascii")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(encoded_dtype).to_bytes(2, "big"))
        digest.update(encoded_dtype)
        digest.update(len(value.shape).to_bytes(1, "big"))
        for dimension in value.shape:
            digest.update(int(dimension).to_bytes(8, "big"))
        digest.update(value.numpy().tobytes(order="C"))
    return f"{IMAGE_GICO_CAUSAL_STATE_NAMESPACE}:{digest.hexdigest()}"


@dataclass(frozen=True, slots=True)
class ImageGICOCausalTrainingConfig:
    updates: int = 2_000
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    gradient_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        for field in ("updates", "batch_size"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        for field in ("learning_rate", "gradient_clip_norm"):
            value = float(getattr(self, field))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive.")
        weight_decay = float(self.weight_decay)
        if not math.isfinite(weight_decay) or weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and nonnegative.")

    def as_payload(self) -> dict[str, object]:
        return {
            "protocol": CAUSAL_TRAINING_CONFIG_PROTOCOL,
            "updates": self.updates,
            "batch_size_cells": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "gradient_clip_norm": self.gradient_clip_norm,
            "optimizer": "AdamW",
            "optimizer_betas": [0.9, 0.999],
            "optimizer_eps": 1e-8,
            "loss": "terminal_weighted_teacher_forced_path_nll",
            "loss_scale_divisor": STICK_ACTION_COUNT,
            "checkpoint_rule": "final_only",
            "label_smoothing": 0.0,
            "auxiliary_losses": [],
        }


@dataclass(frozen=True, slots=True)
class ImageGICOCausalTrainingReport:
    config: dict[str, object]
    model_config: dict[str, object]
    supervision_sha256: str
    model_state_sha256: str
    trainable_parameter_count: int
    initial_batch_nll: float
    final_batch_nll: float
    final_preclip_gradient_norm: float
    sampled_cell_stream_sha256: str
    alias_support_sizes: tuple[int, ...]
    completed_updates: int

    def as_payload(self) -> dict[str, object]:
        return {
            "protocol": CAUSAL_TRAINING_REPORT_PROTOCOL,
            "config": self.config,
            "model_config": self.model_config,
            "supervision_sha256": self.supervision_sha256,
            "model_state_sha256": self.model_state_sha256,
            "trainable_parameter_count": self.trainable_parameter_count,
            "initial_batch_nll": self.initial_batch_nll,
            "final_batch_nll": self.final_batch_nll,
            "final_preclip_gradient_norm": self.final_preclip_gradient_norm,
            "sampled_cell_stream_sha256": self.sampled_cell_stream_sha256,
            "alias_support_sizes": list(self.alias_support_sizes),
            "completed_updates": self.completed_updates,
            "published_checkpoint": "final_only",
        }


@dataclass(frozen=True, slots=True)
class ImageGICOCausalTrainingResult:
    model: ImageGICOCausalTransformer
    path_bank: ImageGICOCausalPathBank
    report: ImageGICOCausalTrainingReport


def path_log_probs_from_teacher_forced_logits(logits: Tensor, target_tokens: Tensor) -> Tensor:
    if logits.ndim < 3 or logits.shape[-2:] != (
        STICK_ACTION_COUNT,
        TOKEN_COUNT,
    ):
        raise ValueError(f"logits must end with [{STICK_ACTION_COUNT}, {TOKEN_COUNT}].")
    if target_tokens.dtype == torch.bool or target_tokens.is_floating_point():
        raise TypeError("target_tokens must use an integer dtype.")
    if target_tokens.shape != logits.shape[:-1] or target_tokens.device != logits.device:
        raise ValueError("target_tokens must match logits without the token axis.")
    selected = torch.gather(
        torch.log_softmax(logits, dim=-1),
        dim=-1,
        index=target_tokens.to(dtype=torch.int64)[..., None],
    ).squeeze(-1)
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("A target token has zero or non-finite probability.")
    return selected.sum(dim=-1)


def terminal_weighted_path_nll(path_log_probs: Tensor, teacher_weights: Tensor) -> Tensor:
    if path_log_probs.ndim != 2 or teacher_weights.shape != path_log_probs.shape:
        raise ValueError("path_log_probs and teacher_weights must be [batch, unique_paths].")
    if teacher_weights.device != path_log_probs.device:
        raise ValueError("teacher_weights and path_log_probs must share a device.")
    if not bool(torch.isfinite(path_log_probs).all()) or not bool(torch.isfinite(teacher_weights).all()):
        raise ValueError("Loss inputs must be finite.")
    if bool(torch.any(teacher_weights < 0.0)):
        raise ValueError("teacher_weights must be nonnegative.")
    totals = teacher_weights.to(dtype=torch.float64).sum(dim=-1)
    if not bool(torch.allclose(totals, torch.ones_like(totals), rtol=0.0, atol=1e-10)):
        raise ValueError("teacher_weights rows must sum to one.")
    return -(teacher_weights.to(dtype=torch.float64) * path_log_probs.to(dtype=torch.float64)).sum(
        dim=-1
    ).mean() / float(STICK_ACTION_COUNT)


def _support_tensors(
    path_bank: ImageGICOCausalPathBank,
    aggregated_weights: tuple[np.ndarray, ...],
    *,
    device: torch.device,
    context_count: int,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...]]:
    paths_out: list[Tensor] = []
    masks_out: list[Tensor] = []
    weights_out: list[Tensor] = []
    for nfe_index, target_nfe in enumerate(TARGET_NFES):
        paths = np.array(
            path_bank.unique_token_paths_by_nfe[nfe_index],
            dtype=np.int64,
            order="C",
            copy=True,
        )
        masks = np.zeros((paths.shape[0], STICK_ACTION_COUNT, TOKEN_COUNT), dtype=np.bool_)
        for path_index, path in enumerate(paths):
            for step in range(STICK_ACTION_COUNT):
                masks[path_index, step] = path_bank.valid_next_token_mask(
                    target_nfe,
                    tuple(int(value) for value in path[:step]),
                )
                if not masks[path_index, step, int(path[step])]:
                    raise ValueError("The support trie rejects one of its own paths.")
        weights = np.array(
            aggregated_weights[nfe_index],
            dtype=np.float64,
            order="C",
            copy=True,
        )
        if weights.shape != (context_count, paths.shape[0]):
            raise ValueError("Aggregated weights and unique paths disagree.")
        paths_out.append(torch.as_tensor(paths, dtype=torch.int64, device=device))
        masks_out.append(torch.as_tensor(masks, dtype=torch.bool, device=device))
        weights_out.append(torch.as_tensor(weights, dtype=torch.float64, device=device))
    return tuple(paths_out), tuple(masks_out), tuple(weights_out)


def train_image_gico_causal_student(
    supervision: ImageGICOSupervision,
    *,
    device: torch.device | str,
    config: ImageGICOCausalTrainingConfig | None = None,
) -> ImageGICOCausalTrainingResult:
    """Train causal-AR from the same finite law as the barycenter student."""

    if not isinstance(supervision, ImageGICOSupervision):
        raise TypeError("supervision must be ImageGICOSupervision.")
    supervision.verify()
    training = ImageGICOCausalTrainingConfig() if config is None else config
    if not isinstance(training, ImageGICOCausalTrainingConfig):
        raise TypeError("config must be ImageGICOCausalTrainingConfig.")
    execution_device, cuda_devices = resolve_image_gico_training_device(device)
    reference = np.linspace(
        0.0,
        1.0,
        supervision.fixed_density_mass.shape[-1] + 1,
        dtype=np.float64,
    )
    path_bank = ImageGICOCausalPathBank.build(
        np.ascontiguousarray(supervision.fixed_density_mass),
        np.ascontiguousarray(reference),
    )
    aggregated = path_bank.aggregate_teacher_weights(np.ascontiguousarray(supervision.mixture_weights))
    context_table = torch.as_tensor(
        np.ascontiguousarray(supervision.normalized_contexts, dtype=np.float32),
        dtype=torch.float32,
        device=execution_device,
    )
    with torch.random.fork_rng(devices=cuda_devices):
        seed_image_gico_training_generators(training.seed, execution_device)
        model = ImageGICOCausalTransformer(ImageGICOCausalConfig()).to(execution_device)
        if model.trainable_parameter_count != EXPECTED_TRAINABLE_PARAMETER_COUNT:
            raise RuntimeError("Causal student parameter count changed.")
        paths, masks, weights = _support_tensors(
            path_bank,
            aggregated,
            device=execution_device,
            context_count=supervision.context_count,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training.learning_rate,
            weight_decay=training.weight_decay,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(training.seed)
        stream_hash = hashlib.sha256()
        first_loss = math.nan
        final_loss = math.nan
        final_gradient = math.nan
        model.train()
        cell_count = len(TARGET_NFES) * supervision.context_count
        for update in range(training.updates):
            cells = torch.randint(
                0,
                cell_count,
                (training.batch_size,),
                generator=generator,
                dtype=torch.int64,
            )
            stream_hash.update(cells.numpy().astype("<i8", copy=False).tobytes(order="C"))
            nfe_indices = torch.div(cells, supervision.context_count, rounding_mode="floor").to(execution_device)
            context_indices = torch.remainder(cells, supervision.context_count).to(execution_device)
            negative_log_probability = torch.zeros((), dtype=torch.float64, device=execution_device)
            for nfe_index, target_nfe in enumerate(TARGET_NFES):
                selected = torch.nonzero(nfe_indices == nfe_index, as_tuple=False).flatten()
                if selected.numel() == 0:
                    continue
                selected_contexts = context_indices[selected]
                path_log_probs = model.enumerate_path_log_probs(
                    context_table[selected_contexts],
                    target_nfe,
                    paths[nfe_index],
                    valid_token_masks=masks[nfe_index],
                )
                negative_log_probability -= (
                    weights[nfe_index][selected_contexts] * path_log_probs.to(dtype=torch.float64)
                ).sum()
            loss = negative_log_probability / float(training.batch_size * STICK_ACTION_COUNT)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Training loss became non-finite at {update}.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=training.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            if update == 0:
                first_loss = float(loss.detach().cpu())
            final_loss = float(loss.detach().cpu())
            final_gradient = float(gradient.detach().cpu())
        model.eval()
    model_state_sha256 = image_gico_causal_state_sha256(model)
    report = ImageGICOCausalTrainingReport(
        config=training.as_payload(),
        model_config=model.config.as_payload(),
        supervision_sha256=supervision.sha256,
        model_state_sha256=model_state_sha256,
        trainable_parameter_count=model.trainable_parameter_count,
        initial_batch_nll=first_loss,
        final_batch_nll=final_loss,
        final_preclip_gradient_norm=final_gradient,
        sampled_cell_stream_sha256=stream_hash.hexdigest(),
        alias_support_sizes=tuple(len(value) for value in path_bank.unique_token_paths_by_nfe),
        completed_updates=training.updates,
    )
    return ImageGICOCausalTrainingResult(model=model, path_bank=path_bank, report=report)


__all__ = [
    "CAUSAL_TRAINING_CONFIG_PROTOCOL",
    "CAUSAL_TRAINING_REPORT_PROTOCOL",
    "IMAGE_GICO_CAUSAL_STATE_NAMESPACE",
    "ImageGICOCausalTrainingConfig",
    "ImageGICOCausalTrainingReport",
    "ImageGICOCausalTrainingResult",
    "image_gico_causal_state_sha256",
    "path_log_probs_from_teacher_forced_logits",
    "terminal_weighted_path_nll",
    "train_image_gico_causal_student",
]
