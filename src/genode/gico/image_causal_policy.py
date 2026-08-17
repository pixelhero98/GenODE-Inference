"""Qualified causal Transformer and strict-support image GICO policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from torch import Tensor, nn

from genode.gico.image_causal_stick import (
    INTERIOR_TOKEN_COUNT,
    STICK_ACTION_COUNT,
    TARGET_NFES,
    TOKEN_COUNT,
    ImageGICOCausalPathBank,
    guard_density_for_inverse_cdf,
    inverse_cdf_clock_nodes,
)

CONTEXT_DIM: Final[int] = 768
D_MODEL: Final[int] = 128
NFE_EMBEDDING_DIM: Final[int] = 16
FFN_DIM: Final[int] = 192
HEAD_COUNT: Final[int] = 4
EXPECTED_TRAINABLE_PARAMETER_COUNT: Final[int] = 339_184


@dataclass(frozen=True, slots=True)
class ImageGICOCausalConfig:
    """Architecture used by the latest authenticated image students."""

    target_nfes: tuple[int, ...] = TARGET_NFES
    action_count: int = STICK_ACTION_COUNT
    token_count: int = TOKEN_COUNT
    context_dim: int = CONTEXT_DIM
    d_model: int = D_MODEL
    nfe_embedding_dim: int = NFE_EMBEDDING_DIM
    head_count: int = HEAD_COUNT
    ffn_dim: int = FFN_DIM
    dropout: float = 0.0
    pre_norm: bool = True
    gelu_approximation: str = "none"

    def __post_init__(self) -> None:
        expected = (
            TARGET_NFES,
            STICK_ACTION_COUNT,
            TOKEN_COUNT,
            CONTEXT_DIM,
            D_MODEL,
            NFE_EMBEDDING_DIM,
            HEAD_COUNT,
            FFN_DIM,
            0.0,
            True,
            "none",
        )
        actual = (
            self.target_nfes,
            self.action_count,
            self.token_count,
            self.context_dim,
            self.d_model,
            self.nfe_embedding_dim,
            self.head_count,
            self.ffn_dim,
            self.dropout,
            self.pre_norm,
            self.gelu_approximation,
        )
        if actual != expected:
            raise ValueError("Image GICO causal architecture is protocol-fixed.")

    def as_payload(self) -> dict[str, object]:
        return {
            "protocol": "gico-causal-transformer-config-v2",
            "target_nfes": list(self.target_nfes),
            "action_count": self.action_count,
            "token_count": self.token_count,
            "context_dim": self.context_dim,
            "d_model": self.d_model,
            "nfe_embedding_dim": self.nfe_embedding_dim,
            "head_count": self.head_count,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
            "pre_norm": self.pre_norm,
            "gelu_approximation": self.gelu_approximation,
            "sequence": "condition_token_then_63_shifted_action_inputs",
            "global_logits_shape": [
                len(TARGET_NFES),
                STICK_ACTION_COUNT,
                TOKEN_COUNT,
            ],
            "residual_head_initialization": "zero",
        }


class ImageGICOCausalTransformer(nn.Module):
    """One-block pre-norm causal Transformer with explicit support masks."""

    def __init__(self, config: ImageGICOCausalConfig | None = None) -> None:
        super().__init__()
        self.config = ImageGICOCausalConfig() if config is None else config
        if not isinstance(self.config, ImageGICOCausalConfig):
            raise TypeError("config must be ImageGICOCausalConfig.")
        self.nfe_embedding = nn.Embedding(len(TARGET_NFES), NFE_EMBEDDING_DIM)
        self.condition_projection = nn.Linear(CONTEXT_DIM + NFE_EMBEDDING_DIM, D_MODEL)
        self.condition_activation = nn.Tanh()
        self.action_embedding = nn.Embedding(TOKEN_COUNT, D_MODEL)
        self.position_embedding = nn.Embedding(STICK_ACTION_COUNT, D_MODEL)
        self.bos = nn.Parameter(torch.zeros(D_MODEL))
        self.remaining_mass_projection = nn.Linear(1, D_MODEL)
        self.block = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=HEAD_COUNT,
            dim_feedforward=FFN_DIM,
            dropout=0.0,
            activation=nn.GELU(approximate="none"),
            batch_first=True,
            norm_first=True,
            bias=True,
        )
        self.global_logits = nn.Parameter(torch.zeros(len(TARGET_NFES), STICK_ACTION_COUNT, TOKEN_COUNT))
        self.residual_head = nn.Linear(D_MODEL, TOKEN_COUNT)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        causal_mask = torch.triu(
            torch.ones(
                STICK_ACTION_COUNT + 1,
                STICK_ACTION_COUNT + 1,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        self.register_buffer("_causal_mask", causal_mask, persistent=False)
        count = sum(parameter.numel() for parameter in self.parameters())
        if count != EXPECTED_TRAINABLE_PARAMETER_COUNT:
            raise RuntimeError(
                "Causal Transformer parameter count drifted: "
                f"expected {EXPECTED_TRAINABLE_PARAMETER_COUNT}, got {count}."
            )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def verify_protocol_buffers(self) -> None:
        expected = torch.triu(
            torch.ones(
                STICK_ACTION_COUNT + 1,
                STICK_ACTION_COUNT + 1,
                dtype=torch.bool,
                device=self._causal_mask.device,
            ),
            diagonal=1,
        )
        if self._causal_mask.dtype != torch.bool or not torch.equal(self._causal_mask, expected):
            raise ValueError("Causal Transformer attention mask was mutated.")

    @property
    def device(self) -> torch.device:
        return self.global_logits.device

    def _validate_contexts(self, contexts: Tensor) -> Tensor:
        if not isinstance(contexts, Tensor):
            raise TypeError("contexts must be a torch.Tensor.")
        if contexts.dtype != torch.float32:
            raise TypeError("contexts must use torch.float32.")
        if contexts.device != self.device:
            raise ValueError("contexts and model must share a device.")
        if contexts.ndim != 2 or contexts.shape[1] != CONTEXT_DIM:
            raise ValueError(f"contexts must have shape [batch, {CONTEXT_DIM}].")
        if contexts.shape[0] <= 0 or not bool(torch.isfinite(contexts).all()):
            raise ValueError("contexts must be nonempty and finite.")
        return contexts

    def _target_nfe_tensor(self, target_nfes: int | Tensor, *, batch_size: int) -> Tensor:
        if isinstance(target_nfes, bool):
            raise TypeError("target_nfes must contain integers.")
        if isinstance(target_nfes, int):
            values = torch.full(
                (batch_size,),
                target_nfes,
                dtype=torch.int64,
                device=self.device,
            )
        elif isinstance(target_nfes, Tensor):
            values = target_nfes
            if values.dtype == torch.bool or values.is_floating_point():
                raise TypeError("target_nfes must use an integer dtype.")
            if values.device != self.device:
                raise ValueError("target_nfes and model must share a device.")
            if values.ndim != 1 or values.shape[0] != batch_size:
                raise ValueError("target_nfes must have shape [batch].")
        else:
            raise TypeError("target_nfes must be an int or torch.Tensor.")
        return values

    def _nfe_indices(self, target_nfes: int | Tensor, *, batch_size: int) -> Tensor:
        values = self._target_nfe_tensor(target_nfes, batch_size=batch_size)
        supported = torch.tensor(TARGET_NFES, dtype=values.dtype, device=values.device)
        matches = values[:, None] == supported[None, :]
        if not bool(torch.all(matches.sum(dim=1) == 1)):
            raise ValueError(f"target_nfes must contain only {TARGET_NFES}.")
        return torch.argmax(matches.to(dtype=torch.int64), dim=1)

    def _validate_token_paths(self, token_paths: Tensor, *, batch_size: int) -> Tensor:
        if not isinstance(token_paths, Tensor):
            raise TypeError("token_paths must be a torch.Tensor.")
        if token_paths.device != self.device:
            raise ValueError("token_paths and model must share a device.")
        if token_paths.dtype == torch.bool or token_paths.is_floating_point():
            raise TypeError("token_paths must use an integer dtype.")
        if tuple(token_paths.shape) != (batch_size, STICK_ACTION_COUNT):
            raise ValueError(f"token_paths must have shape [batch, {STICK_ACTION_COUNT}].")
        if bool(torch.any((token_paths < 0) | (token_paths >= TOKEN_COUNT))):
            raise ValueError(f"token_paths must lie in [0, {TOKEN_COUNT - 1}].")
        return token_paths.to(dtype=torch.int64)

    @staticmethod
    def remaining_masses_from_tokens(token_paths: Tensor) -> Tensor:
        if token_paths.dtype == torch.bool or token_paths.is_floating_point():
            raise TypeError("token_paths must use an integer dtype.")
        if token_paths.ndim != 2 or token_paths.shape[1] != STICK_ACTION_COUNT:
            raise ValueError(f"token_paths must have shape [batch, {STICK_ACTION_COUNT}].")
        if bool(torch.any((token_paths < 0) | (token_paths >= TOKEN_COUNT))):
            raise ValueError(f"token_paths must lie in [0, {TOKEN_COUNT - 1}].")
        values = token_paths.to(dtype=torch.float64)
        interior = ((values - 1.0 + 0.5) / float(INTERIOR_TOKEN_COUNT)).pow(3)
        actions = torch.where(
            token_paths == 0,
            torch.zeros_like(interior),
            torch.where(
                token_paths == TOKEN_COUNT - 1,
                torch.ones_like(interior),
                interior,
            ),
        )
        leading = torch.ones(
            token_paths.shape[0],
            1,
            dtype=torch.float64,
            device=token_paths.device,
        )
        return torch.cat((leading, torch.cumprod((1.0 - actions)[:, :-1], dim=1)), dim=1)

    def teacher_forced_logits(
        self,
        contexts: Tensor,
        target_nfes: int | Tensor,
        token_paths: Tensor,
        *,
        valid_token_masks: Tensor | None,
    ) -> Tensor:
        self.verify_protocol_buffers()
        contexts = self._validate_contexts(contexts)
        paths = self._validate_token_paths(token_paths, batch_size=contexts.shape[0])
        nfe_indices = self._nfe_indices(target_nfes, batch_size=contexts.shape[0])
        masks = valid_token_masks
        if masks is not None:
            expected = (contexts.shape[0], STICK_ACTION_COUNT, TOKEN_COUNT)
            if (
                not isinstance(masks, Tensor)
                or masks.dtype != torch.bool
                or masks.device != self.device
                or tuple(masks.shape) != expected
                or not bool(masks.any(dim=-1).all())
            ):
                raise ValueError(f"valid_token_masks must be a same-device bool tensor {expected}.")
            targets_are_valid = torch.gather(masks, dim=-1, index=paths[..., None]).squeeze(-1)
            if not bool(targets_are_valid.all()):
                raise ValueError("A target token is outside the support mask.")

        condition = self.condition_activation(
            self.condition_projection(torch.cat((contexts, self.nfe_embedding(nfe_indices)), dim=-1))
        )
        shifted = self.bos.view(1, 1, D_MODEL).expand(contexts.shape[0], 1, D_MODEL)
        shifted = torch.cat((shifted, self.action_embedding(paths[:, :-1])), dim=1)
        positions = self.position_embedding(torch.arange(STICK_ACTION_COUNT, device=self.device))
        remaining = self.remaining_masses_from_tokens(paths).to(dtype=contexts.dtype)
        action_inputs = shifted + positions.unsqueeze(0) + self.remaining_mass_projection(remaining.unsqueeze(-1))
        sequence = torch.cat((condition.unsqueeze(1), action_inputs), dim=1)
        encoded = self.block(sequence, src_mask=self._causal_mask)
        logits = self.global_logits[nfe_indices] + self.residual_head(encoded[:, 1:, :])
        return logits if masks is None else logits.masked_fill(~masks, -torch.inf)

    def enumerate_path_log_probs(
        self,
        contexts: Tensor,
        target_nfes: int | Tensor,
        path_tokens: Tensor,
        *,
        valid_token_masks: Tensor,
    ) -> Tensor:
        contexts = self._validate_contexts(contexts)
        if (
            not isinstance(path_tokens, Tensor)
            or path_tokens.device != self.device
            or path_tokens.dtype == torch.bool
            or path_tokens.is_floating_point()
            or path_tokens.ndim != 2
            or path_tokens.shape[1] != STICK_ACTION_COUNT
        ):
            raise ValueError(f"path_tokens must be a same-device integer [paths, {STICK_ACTION_COUNT}] tensor.")
        path_count = path_tokens.shape[0]
        expected_masks = (path_count, STICK_ACTION_COUNT, TOKEN_COUNT)
        if (
            not isinstance(valid_token_masks, Tensor)
            or valid_token_masks.dtype != torch.bool
            or valid_token_masks.device != self.device
            or tuple(valid_token_masks.shape) != expected_masks
        ):
            raise ValueError(f"valid_token_masks must have shape {expected_masks}.")
        target_values = self._target_nfe_tensor(target_nfes, batch_size=contexts.shape[0])
        if not bool(torch.all(target_values == target_values[0])):
            raise ValueError("Path enumeration requires one NFE-specific support.")
        batch_size = contexts.shape[0]
        expanded_contexts = (
            contexts[:, None, :]
            .expand(batch_size, path_count, CONTEXT_DIM)
            .reshape(batch_size * path_count, CONTEXT_DIM)
        )
        expanded_nfes = target_values[:, None].expand(batch_size, path_count).reshape(batch_size * path_count)
        expanded_paths = (
            path_tokens[None]
            .expand(batch_size, path_count, STICK_ACTION_COUNT)
            .reshape(batch_size * path_count, STICK_ACTION_COUNT)
        )
        expanded_masks = (
            valid_token_masks[None]
            .expand(batch_size, path_count, STICK_ACTION_COUNT, TOKEN_COUNT)
            .reshape(batch_size * path_count, STICK_ACTION_COUNT, TOKEN_COUNT)
        )
        logits = self.teacher_forced_logits(
            expanded_contexts,
            expanded_nfes,
            expanded_paths,
            valid_token_masks=expanded_masks,
        )
        selected = torch.gather(
            torch.log_softmax(logits, dim=-1),
            dim=-1,
            index=expanded_paths[..., None],
        ).squeeze(-1)
        return selected.sum(dim=-1).reshape(batch_size, path_count)


@dataclass(frozen=True, slots=True)
class FrozenImageGICOCausalRealization:
    target_nfe: int
    tokens: tuple[tuple[int, ...], ...]
    raw_densities: tuple[tuple[float, ...], ...]
    guarded_densities: tuple[tuple[float, ...], ...]
    time_grids: tuple[tuple[float, ...], ...]
    uniforms_consumed_per_image: int = STICK_ACTION_COUNT


class ImageGICOCausalPolicy:
    """Sample a complete supported path before Euler execution starts."""

    def __init__(
        self,
        model: ImageGICOCausalTransformer,
        path_bank: ImageGICOCausalPathBank,
    ) -> None:
        if not isinstance(model, ImageGICOCausalTransformer):
            raise TypeError("model must be ImageGICOCausalTransformer.")
        if not isinstance(path_bank, ImageGICOCausalPathBank):
            raise TypeError("path_bank must be ImageGICOCausalPathBank.")
        self.model = model
        self.path_bank = path_bank
        self.reference_time_grid = np.array(
            path_bank.reference_time_grid,
            dtype=np.float64,
            order="C",
            copy=True,
        )
        self.reference_time_grid.setflags(write=False)

    def sample_realization(
        self,
        contexts: Tensor,
        target_nfe: int,
        uniforms: Tensor,
    ) -> FrozenImageGICOCausalRealization:
        contexts = self.model._validate_contexts(contexts)
        if isinstance(target_nfe, bool) or target_nfe not in TARGET_NFES:
            raise ValueError(f"target_nfe must be one of {TARGET_NFES}.")
        expected = (contexts.shape[0], STICK_ACTION_COUNT)
        if (
            not isinstance(uniforms, Tensor)
            or uniforms.device.type != "cpu"
            or uniforms.dtype != torch.float64
            or tuple(uniforms.shape) != expected
        ):
            raise TypeError(f"uniforms must be a CPU float64 tensor {expected}.")
        if not bool(torch.isfinite(uniforms).all()) or bool(torch.any((uniforms <= 0.0) | (uniforms >= 1.0))):
            raise ValueError("uniforms must lie strictly inside (0, 1).")

        realized = torch.zeros(
            contexts.shape[0],
            STICK_ACTION_COUNT,
            dtype=torch.int64,
            device=self.model.device,
        )
        with torch.inference_mode():
            for step in range(STICK_ACTION_COUNT):
                logits = self.model.teacher_forced_logits(
                    contexts,
                    target_nfe,
                    realized,
                    valid_token_masks=None,
                )[:, step]
                masks_np = np.stack(
                    [
                        self.path_bank.valid_next_token_mask(
                            target_nfe,
                            tuple(int(value) for value in realized[row, :step].detach().cpu().tolist()),
                        )
                        for row in range(contexts.shape[0])
                    ]
                )
                masks = torch.as_tensor(masks_np, dtype=torch.bool, device=self.model.device)
                probabilities = torch.softmax(
                    logits.masked_fill(~masks, -torch.inf).detach().to(device="cpu", dtype=torch.float64),
                    dim=-1,
                )
                cumulative = torch.cumsum(probabilities, dim=-1)
                cumulative[:, -1] = 1.0
                next_tokens = torch.searchsorted(
                    cumulative.contiguous(),
                    uniforms[:, step, None].contiguous(),
                    right=False,
                ).squeeze(-1)
                if not bool(
                    torch.gather(
                        torch.as_tensor(masks_np, dtype=torch.bool),
                        dim=-1,
                        index=next_tokens[:, None],
                    ).all()
                ):
                    raise RuntimeError("Sampling selected an unsupported token.")
                realized[:, step] = next_tokens.to(device=self.model.device)

        token_array = realized.detach().cpu().numpy().astype(np.int64, copy=False)
        raw_density = np.ascontiguousarray(
            np.stack([self.path_bank.decode_supported_path(target_nfe, row) for row in token_array]),
            dtype=np.float64,
        )
        guarded = guard_density_for_inverse_cdf(raw_density)
        grids = inverse_cdf_clock_nodes(
            raw_density,
            target_nfe,
            self.reference_time_grid,
        )
        return FrozenImageGICOCausalRealization(
            target_nfe=target_nfe,
            tokens=tuple(tuple(int(value) for value in row) for row in token_array),
            raw_densities=tuple(tuple(float(value) for value in row) for row in raw_density),
            guarded_densities=tuple(tuple(float(value) for value in row) for row in guarded),
            time_grids=tuple(tuple(float(value) for value in row) for row in grids),
        )


__all__ = [
    "CONTEXT_DIM",
    "D_MODEL",
    "EXPECTED_TRAINABLE_PARAMETER_COUNT",
    "FFN_DIM",
    "FrozenImageGICOCausalRealization",
    "HEAD_COUNT",
    "ImageGICOCausalConfig",
    "ImageGICOCausalPolicy",
    "ImageGICOCausalTransformer",
    "NFE_EMBEDDING_DIM",
]
